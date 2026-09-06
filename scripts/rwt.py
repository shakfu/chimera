#!/usr/bin/env python3
"""Self-contained smoke-test runner for released chimera binaries.

chimera ships as one static executable, so there is nothing to `pip install`
and no virtualenv to reason about: ``install`` downloads a release archive
from GitHub and unpacks the binary into ``build/rwt/``, and every test target runs
that binary's CLI. ``--bin`` points at an executable somewhere else (a local
``build/chimera``, say) when the thing under test is not a release artifact.

``--cuda`` (and ``--cpu`` / ``--metal`` / ``--vulkan`` / ``--rocm`` /
``--sycl``) names the backend, which selects the release asset for this
platform -- ``--cuda`` on Linux is ``chimera-<version>-linux-x86_64-cuda.tar.gz``.
Without one the backend is read back out of the installed binary's own
``chimera info`` (`built:`), so a test run never has to be told what it is
testing. ``--cpu`` and ``--metal`` name the same macOS asset: CI builds the
macos-arm64 artifact with Metal on and there is no CPU-only macOS build, so a
plain macos-arm64 binary is reported as ``metal``.

``install`` is the only subcommand that writes to ``build/rwt/``: ``--asset`` says
what to put there -- a local archive, a full URL, or a bare release-asset
filename, told apart by shape -- and ``--version`` picks the release when it
does not (default: whatever ``/releases/latest`` resolves to). Every test
target expects a binary that is already in place.

``test`` takes one target -- ``test-all``, ``test-gen-all``, ``test-sd-3`` --
named identically to the generated Makefile rules; ``list tests`` prints them.

``run`` is ``install``, ``test`` and ``clean`` in one invocation, stopping at
the first step that fails and taking the options of all three. It is the whole
cycle for one backend, so a release can be checked on a machine that has
nothing installed yet without three commands that must agree on which binary
they mean. ``--fast`` swaps ``test-all`` for ``test-embed-1``, ``test-gen-1``
and ``test-sd-4`` -- the same shape of coverage without the image cases that
dominate the wall clock, or the one case whose model may not be downloadable.

The cases are the shell scripts that used to live in ``scripts/case/``,
inlined here so they share one model registry, one timeout, one summary and
one place that knows a GPU build needs ``--gpu-layers``.

The script is organised as a handful of collaborating objects rather than
module state: :class:`Paths` resolves the directory layout, :class:`Release`
knows how release assets are named and fetched, :class:`Env` owns the binary
under test, :class:`ModelRegistry` knows where models and data assets come
from, :class:`TestSuite` holds the test cases, and :class:`Cli` wires them to
argparse.

Examples:
    # download the latest linux-x86_64-cuda release into ./bin and test it
    python3 scripts/rwt.py install --cuda
    python3 scripts/rwt.py test --cuda test-all

    # a specific release, or a local artifact / explicit URL
    python3 scripts/rwt.py install --cuda --version 0.2.16
    python3 scripts/rwt.py install --asset dist/chimera-0.2.16-linux-x86_64-cuda.tar.gz
    python3 scripts/rwt.py install --asset https://github.com/shakfu/chimera/releases/download/0.2.16/chimera-0.2.16-linux-x86_64-cuda.tar.gz

    # install, test everything, then remove the binary again -- one command
    python3 scripts/rwt.py run --cuda
    python3 scripts/rwt.py run --cuda --fast    # a short cycle instead of everything
    python3 scripts/rwt.py run --vulkan test-sd-all --timeout 900

    # run everything, one family, or one case
    python3 scripts/rwt.py test test-all
    python3 scripts/rwt.py test test-rag-all
    python3 scripts/rwt.py test test-sd-4 --timeout 600

    # against a binary that was not installed from a release; the backend is
    # detected from `chimera info`, so no --cuda/--vulkan/... is needed
    python3 scripts/rwt.py test --bin build/chimera test-all

    # show the matrix without downloading or running anything
    python3 scripts/rwt.py test --cuda test-all --dry-run

    # environment, registry and target listings
    python3 scripts/rwt.py info
    python3 scripts/rwt.py list
    python3 scripts/rwt.py download all --models-dir models
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_NAME = Path(__file__).name


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


class ModelSourceUnavailable(RuntimeError):
    """Raised when a model has no configured source and isn't on disk."""


class AssetUnavailable(RuntimeError):
    """Raised when no release asset exists for a backend/platform pair."""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    """The directory layout every other object resolves against."""

    root: Path
    # `build/` rather than the project root as the base for everything this
    # script creates: it is already gitignored and already the directory
    # `make clean` sweeps -- a smoke run should not leave anything for the next
    # `git status` to report, and throwing away a tested binary along with the
    # build tree is the right default. `models/` is the deliberate exception:
    # it stays at the root, shared with `make test`, because re-downloading
    # tens of GiB of weights after every `make clean` is not.
    build_dir: Path
    # One directory under it holds this script's whole footprint -- the
    # installed binary at `<rwt_dir>/chimera`, everything a run produces in
    # `<rwt_dir>/out` -- so it is obvious at a glance what rwt.py owns inside a
    # build tree it shares with cmake.
    rwt_dir: Path
    models_dir: Path
    data_dir: Path
    bin_dir: Path
    out_dir: Path

    @staticmethod
    def find_root() -> Path:
        """Locate the project root: the cwd for subprocesses, the parent of
        ``models/``, and what ``build/`` is resolved against.

        This file is checked in as ``<repo>/scripts/rwt.py`` but is also meant
        to be copied out standalone (as ``./rwt.py``) into a bare directory
        that holds nothing but ``models/`` and a ``build/rwt/``. Walking up to the
        nearest project marker handles both layouts; using ``__file__``'s own
        directory would resolve to ``<repo>/scripts`` in-repo and download
        models to ``scripts/models``.
        """
        here = Path(__file__).resolve().parent
        for candidate in (here, *here.parents):
            if (candidate / "CMakeLists.txt").exists() or (candidate / ".git").exists():
                return candidate
        return here

    @staticmethod
    def resolve_build_dir(root: Path) -> Path:
        """Where `build/` is. Honours ``BUILD_DIR`` the way the Makefile does
        (``BUILD_DIR ?= build``), so an out-of-tree build directory is named
        once and both agree on it."""
        raw = os.environ.get("CHIMERA_BUILD_DIR") or os.environ.get("BUILD_DIR") or "build"
        build = Path(raw).expanduser()
        return build if build.is_absolute() else root / build

    @classmethod
    def from_environ(cls) -> Paths:
        root = cls.find_root()
        build = cls.resolve_build_dir(root)
        rwt = Path(os.environ.get("CHIMERA_RWT_DIR", build / "rwt"))
        return cls(
            root=root,
            build_dir=build,
            rwt_dir=rwt,
            models_dir=Path(os.environ.get("CHIMERA_MODELS_DIR", root / "models")),
            data_dir=Path(os.environ.get("CHIMERA_DATA_DIR", build / "whisper.cpp" / "samples")),
            bin_dir=Path(os.environ.get("CHIMERA_BIN_DIR", rwt)),
            out_dir=Path(os.environ.get("CHIMERA_RWT_OUT", rwt / "out")),
        )

    def rebase(self, build: Path) -> None:
        """Re-point every build-relative default at `build`.

        Called when --build-dir moves it after construction. Only the defaults
        move: a directory the caller named outright with --bin-dir / --out-dir /
        --data-dir is applied afterwards and wins.
        """
        old = self.build_dir
        self.build_dir = build
        for attr in ("rwt_dir", "data_dir", "bin_dir", "out_dir"):
            current = getattr(self, attr)
            try:
                setattr(self, attr, build / current.relative_to(old))
            except ValueError:
                pass  # explicitly set elsewhere (env var); leave it alone

    @property
    def cache_dir(self) -> Path:
        """Where downloaded release archives are kept.

        A sibling of `out_dir`, not a child: an archive is the *input* to a run
        -- the artifact under test -- while `out/` is what a run produced.
        """
        return self.rwt_dir / "downloads"

    @property
    def data_dirs(self) -> list[Path]:
        # jfk.wav is not checked in: it arrives under <build>/whisper.cpp/samples
        # when `make deps` fetches the vendored whisper.cpp tree. Standalone
        # there is no such tree, and ModelRegistry downloads it instead.
        return [
            self.data_dir,
            self.build_dir / "whisper.cpp" / "samples",
            self.root / "tests" / "media",
        ]

    def find_data_asset(self, name: str) -> Path | None:
        """First existing copy of `name` in --data-dir or the checkout's data dirs."""
        for d in self.data_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
        return None


# ---------------------------------------------------------------------------
# release assets
# ---------------------------------------------------------------------------


class Release:
    """How chimera's release assets are named, resolved and unpacked.

    The names come straight from the two release workflows: `release.yml`
    stages ``chimera-<version>-<target>.{tar.gz,zip}`` for the CPU/Metal
    matrix, and `release-gpu.yml` appends the backend to the target for each
    GPU leg (``linux-x86_64-cuda``, ``windows-x86_64-vulkan``, ...). Each
    archive contains exactly one member: the bare ``chimera`` binary.
    """

    DEFAULT_REPO = "shakfu/chimera"
    API = "https://api.github.com/repos/{repo}/releases/{ref}"
    DOWNLOAD = "https://github.com/{repo}/releases/download/{tag}/{asset}"

    # (backend, os, arch) -> release target. Only the pairs CI actually
    # publishes are here; anything else is an AssetUnavailable rather than a
    # 404 halfway through a download. `cpu` and `metal` share the macOS entry
    # because there is no CPU-only macOS build to tell them apart.
    TARGETS: dict[tuple[str, str, str], str] = {
        ("cpu", "linux", "x86_64"): "linux-x86_64",
        ("cpu", "windows", "x86_64"): "windows-x86_64",
        ("cpu", "macos", "arm64"): "macos-arm64",
        ("metal", "macos", "arm64"): "macos-arm64",
        ("cuda", "linux", "x86_64"): "linux-x86_64-cuda",
        ("cuda", "windows", "x86_64"): "windows-x86_64-cuda",
        ("vulkan", "linux", "x86_64"): "linux-x86_64-vulkan",
        ("vulkan", "windows", "x86_64"): "windows-x86_64-vulkan",
        ("rocm", "linux", "x86_64"): "linux-x86_64-rocm",
        ("sycl", "linux", "x86_64"): "linux-x86_64-sycl",
    }

    def __init__(self, repo: str | None = None) -> None:
        self.repo = repo or os.environ.get("CHIMERA_RWT_REPO", self.DEFAULT_REPO)

    # -- naming -------------------------------------------------------------

    @staticmethod
    def host() -> tuple[str, str]:
        """(os, arch) in the spelling :attr:`TARGETS` uses."""
        system = {"Darwin": "macos", "Windows": "windows", "Linux": "linux"}.get(platform.system(), "linux")
        machine = platform.machine().lower()
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        return system, arch

    @classmethod
    def target_for(cls, backend: str) -> str:
        system, arch = cls.host()
        try:
            return cls.TARGETS[(backend, system, arch)]
        except KeyError:
            published = sorted({b for b, o, a in cls.TARGETS if (o, a) == (system, arch)})
            raise AssetUnavailable(
                f"no '{backend}' release is published for {system}-{arch} "
                f"(available here: {', '.join(published) or 'none'})"
            ) from None

    @classmethod
    def asset_name(cls, backend: str, version: str) -> str:
        target = cls.target_for(backend)
        # Windows gets a .zip (what Windows users expect); everyone else a
        # .tar.gz -- matching how the workflows package each leg.
        ext = "zip" if target.startswith("windows") else "tar.gz"
        return f"chimera-{version}-{target}.{ext}"

    # -- resolution ---------------------------------------------------------

    def _api(self, ref: str) -> dict[str, Any]:
        url = self.API.format(repo=self.repo, ref=ref)
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        # Anonymous API calls are rate-limited to 60/hour per IP, which a CI
        # matrix can exhaust. Use a token when the environment offers one.
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    def latest_tag(self) -> str:
        try:
            return str(self._api("latest")["tag_name"])
        except (urllib.error.URLError, KeyError, ValueError) as e:
            raise AssetUnavailable(f"could not resolve the latest release of {self.repo}: {e}") from None

    def url_for(self, backend: str, version: str, tag: str | None = None) -> str:
        # Tags have been pushed both as `0.2.16` and `v0.2.16`; the workflows
        # strip a leading `v` for the filename, so the tag and the version in
        # the asset name are not interchangeable. Keep them separate.
        return self.DOWNLOAD.format(repo=self.repo, tag=tag or version, asset=self.asset_name(backend, version))

    # -- fetch + unpack -----------------------------------------------------

    @staticmethod
    def download(url: str, dest: Path) -> Path:
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        tmp.replace(dest)
        return dest

    @staticmethod
    def extract(archive: Path, bin_dir: Path) -> Path:
        """Unpack the single ``chimera`` member of `archive` into `bin_dir`.

        The member is looked up by basename rather than by index: the archives
        are flat today, but a future one that adds a README should still put
        the binary in the right place instead of unpacking whatever came first.
        """
        wanted = {"chimera", "chimera.exe"}
        bin_dir.mkdir(parents=True, exist_ok=True)

        if archive.suffix == ".zip" or zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                names = [n for n in zf.namelist() if Path(n).name in wanted]
                if not names:
                    raise AssetUnavailable(f"{archive.name} contains no chimera binary")
                name = names[0]
                dest = bin_dir / Path(name).name
                with zf.open(name) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
        else:
            with tarfile.open(archive) as tf:
                members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name in wanted]
                if not members:
                    raise AssetUnavailable(f"{archive.name} contains no chimera binary")
                member = members[0]
                dest = bin_dir / Path(member.name).name
                src = tf.extractfile(member)
                if src is None:
                    raise AssetUnavailable(f"could not read {member.name} from {archive.name}")
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

        # tar/zip modes are not carried over by the streaming copy above, and
        # an archive built on Windows has no POSIX mode to carry anyway.
        dest.chmod(dest.stat().st_mode | 0o755)
        print(f"installed {dest}")
        return dest


# ---------------------------------------------------------------------------
# the environment under test
# ---------------------------------------------------------------------------


class Env:
    """The chimera binary under test: where it lives, what it was built with,
    and how subprocesses are run against it.

    There is no virtualenv and no interpreter indirection -- the thing under
    test is one static executable, so `chimera` is invoked directly and the
    only question is *which* executable. ``bin_path`` answers that: an explicit
    ``--bin``, or ``<bin_dir>/chimera`` where ``install`` puts it.
    """

    BACKENDS: tuple[str, ...] = ("cpu", "metal", "cuda", "vulkan", "rocm", "sycl")

    # `chimera info` prints CHIMERA_BUILT_BACKENDS on its `built:` line, using
    # the labels CMakeLists.txt assigns to each GGML_* option. Map them back to
    # this script's backend names. BLAS is a CPU accelerator, not a backend a
    # release is cut for, so it reads as `cpu`.
    BUILT_LABELS: dict[str, str] = {
        "Metal": "metal",
        "CUDA": "cuda",
        "Vulkan": "vulkan",
        "HIP": "rocm",
        "SYCL": "sycl",
        "CPU": "cpu",
        "BLAS": "cpu",
    }

    # chimera's llama-side subcommands default to --gpu-layers 0, i.e. CPU,
    # even in a GPU build -- unlike `sd`, which picks up the GPU on its own.
    # A GPU release tested without this would pass every case while measuring
    # nothing but the CPU path, so the backend chooses the default.
    GPU_LAYERS_DEFAULT = 99

    def __init__(self, paths: Paths, bin_path: Path | None = None, gpu_layers: int | None = None) -> None:
        self.paths = paths
        # Explicit --bin; None means <bin_dir>/chimera.
        self._bin_path = bin_path
        # Explicit --gpu-layers; None means "derive from the backend".
        self.gpu_layers_override = gpu_layers

    # -- the binary ---------------------------------------------------------

    @property
    def exe_name(self) -> str:
        return "chimera.exe" if os.name == "nt" else "chimera"

    @property
    def bin_override(self) -> Path | None:
        """The `--bin` path, or None when the binary is the installed one.

        The distinction matters to `clean`, which may only delete a binary this
        script put there.
        """
        return self._bin_path

    @property
    def bin_path(self) -> Path:
        return self._bin_path if self._bin_path is not None else self.paths.bin_dir / self.exe_name

    @bin_path.setter
    def bin_path(self, value: Path | None) -> None:
        self._bin_path = value

    def gpu_layers(self, backend: str) -> int:
        if self.gpu_layers_override is not None:
            return self.gpu_layers_override
        return 0 if backend == "cpu" else self.GPU_LAYERS_DEFAULT

    def gpu_args(self, backend: str) -> list[str]:
        """``--gpu-layers N`` for the subcommands that take it."""
        return ["--gpu-layers", str(self.gpu_layers(backend))]

    # -- subprocesses -------------------------------------------------------

    @staticmethod
    def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
        """Kill `proc` and every process it spawned.

        chimera is a single process rather than a re-execing launcher, so this
        is usually the same as ``proc.kill()`` -- but a timed-out image run
        that leaves anything behind holds several GiB of VRAM, and every later
        test in the matrix then OOMs or crawls, which silently invalidates the
        whole run's timings. Take the entire tree down instead.
        """
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        else:
            import signal

            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"warning: could not fully reap pid {proc.pid}", file=sys.stderr)

    def run(
        self,
        cmd: list[str],
        env: dict[str, str] | None = None,
        check: bool = False,
        timeout: float | None = None,
    ) -> int:
        """Run a subprocess; return the exit code.

        `check=False` is the default so callers can accumulate failures across
        a smoke-test matrix. Pass ``check=True`` for fail-fast behaviour.
        """
        print(f"$ {' '.join(cmd)}", flush=True)
        full_env = os.environ.copy()
        # Redirected stdout on Windows defaults to the ANSI codepage, and the
        # sd log callback emits byte-level BPE markers (U+0120, U+010A) that
        # cp1252 cannot encode. Force UTF-8 so a logged run matches a console one.
        full_env.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            full_env.update(env)
        proc = subprocess.Popen(cmd, cwd=self.paths.root, env=full_env, start_new_session=os.name != "nt")
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"error: command timed out after {timeout}s", file=sys.stderr)
            self._kill_tree(proc)
            rc = 124  # conventional timeout exit code
        if check and rc != 0:
            sys.exit(rc)
        return rc

    def chimera(self, argv: list[str], env: dict[str, str] | None = None, timeout: float | None = None) -> int:
        return self.run([str(self.bin_path), *argv], env=env, timeout=timeout)

    def capture(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        """Run chimera and capture its output; for probes, not for test cases."""
        return subprocess.run(
            [str(self.bin_path), *argv],
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            errors="replace",
        )

    # -- backend detection --------------------------------------------------

    def info_text(self) -> str | None:
        """``chimera info`` output, or None if the binary will not run."""
        if not self.bin_path.exists():
            return None
        proc = self.capture(["info"])
        return proc.stdout if proc.returncode == 0 else None

    @classmethod
    def parse_info(cls, text: str) -> tuple[str | None, str | None, str | None]:
        """(version, built backend, loaded backend) out of `chimera info`.

        `built:` is a compile-time constant baked into the binary and may be a
        comma-separated list (``Vulkan,BLAS``); the first entry that names a
        real backend wins. `loaded:` is what the ggml registry actually brought
        up on this host, and the two disagreeing is the interesting case -- a
        CUDA build on a box with no driver reports `built: CUDA, loaded: CPU`.
        """
        version = None
        if m := re.match(r"chimera\s+(\S+)", text):
            version = m.group(1)

        built = None
        if m := re.search(r"^\s*built:\s*(.+)$", text, re.M):
            for label in (s.strip() for s in m.group(1).split(",")):
                mapped = cls.BUILT_LABELS.get(label)
                if mapped and (built is None or built == "cpu"):
                    built = mapped

        loaded = None
        if m := re.search(r"^\s*loaded:\s*(.+)$", text, re.M):
            loaded = cls.BUILT_LABELS.get(m.group(1).strip())

        return version, built, loaded

    def detect(self) -> tuple[str | None, str | None, str | None]:
        text = self.info_text()
        return self.parse_info(text) if text else (None, None, None)

    def detect_backend(self) -> str | None:
        return self.detect()[1]

    def require_backend(self, requested: str | None) -> str:
        detected = self.detect_backend()
        if requested and detected and requested != detected:
            print(
                f"warning: requested backend '{requested}' but {self.bin_path} was built for '{detected}'",
                file=sys.stderr,
            )
        backend = requested or detected
        if not backend:
            flags = ",".join("--" + b for b in self.BACKENDS)
            print(
                f"error: no chimera binary at {self.bin_path}."
                f"\n  Install from a release: {SCRIPT_NAME} install {{{flags}}}"
                f"\n  ...or a local archive:  {SCRIPT_NAME} install --asset <path-or-url>"
                f"\n  ...or test a binary you already have: {SCRIPT_NAME} test --bin <path> ...",
                file=sys.stderr,
            )
            sys.exit(2)
        return backend

    def preflight(self, backend: str) -> str | None:
        """Run the binary once up front; return an error message, or None.

        A release archive built for the wrong glibc, or missing a runtime the
        host does not have, fails identically on every case in the matrix.
        Finding that out once, with the loader's own message attached, beats
        watching twelve cases die with the same opaque exit code.
        """
        if not self.bin_path.exists():
            return f"no chimera binary at {self.bin_path}"
        proc = self.capture(["info"])
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            tail = detail[-1] if detail else f"exit code {proc.returncode}"
            return f"cannot run {self.bin_path}: {tail}"

        _version, built, loaded = self.parse_info(proc.stdout)
        if built and built != "cpu" and loaded == "cpu":
            # Not fatal: the suite still runs, just on the CPU. Say so loudly,
            # because otherwise a green matrix looks like the GPU release works.
            print(
                f"warning: {self.bin_path} was built for '{built}' but the ggml registry loaded only CPU"
                f"\n  every case below will run on the CPU; check the driver / runtime for '{built}'",
                file=sys.stderr,
            )
        elif backend != "cpu" and loaded == "cpu":
            print(f"warning: '{backend}' requested but chimera info reports loaded: CPU", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSource:
    """Where to fetch a model from.

    One of repo_id (HF Hub) or url (direct http) must be set.
    """

    filename: str
    repo_id: str | None = None
    hf_filename: str | None = None  # defaults to filename
    url: str | None = None
    notes: str = ""

    def hub_filename(self) -> str:
        return self.hf_filename or self.filename


class ModelRegistry:
    """Known models and data assets, and how to get them onto disk."""

    JFK_WAV_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"

    # Which tests need which models.
    SD_REQUIREMENTS: list[str] = ["z-image-turbo", "ae", "qwen3-4b"]
    RAG_REQUIREMENTS: list[str] = ["bge-small-en"]

    # One text per line -- the format `chimera index ingest -f` and the
    # per-line embed case both read. Deliberately includes a cluster about
    # mortality and one proper noun (Kilimanjaro), so the semantic and the
    # lexical retrieval legs each have something to rank.
    GENERATED_CORPUS: list[str] = [
        "The old man knew that he was dying, and he felt no fear of it.",
        "Death comes for everyone eventually, and grief is the price of having loved.",
        "Mourners gathered at the graveside in the cold morning air.",
        "He had spent his last years writing about mortality and the end of life.",
        "The hospice nurse spoke gently about what the final days would be like.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The compiler performs constant folding before emitting machine code.",
        "Mount Kilimanjaro is the highest free-standing mountain in the world.",
        "She sold the bakery and moved to a small town near the coast.",
        "Quicksort has an average time complexity of O(n log n).",
        "The bridge was rebuilt after the flood washed away its central span.",
        "A leopard was found frozen near the western summit of the mountain.",
        "Offloading model weights to the CPU trades throughput for VRAM headroom.",
    ]

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.sources = self.default_sources()
        self.apply_env_overrides()

    @staticmethod
    def default_sources() -> dict[str, ModelSource]:
        """Best-effort defaults -- overridable via CHIMERA_MODEL_<KEY>=repo_id:file
        or by placing files in the models dir yourself. Use `list models` to inspect.
        """
        return {
            "llama-3.2-1b": ModelSource(
                filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                url="https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf",
            ),
            "qwen3-4b": ModelSource(
                filename="Qwen3-4B-Q8_0.gguf",
                repo_id="Qwen/Qwen3-4B-GGUF",
                url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
            ),
            "gemma-e4b": ModelSource(
                filename="gemma-4-E4B-it-Q5_K_M.gguf",
                repo_id="",  # override via env if/when available
                notes="set CHIMERA_MODEL_GEMMA_E4B=<repo_id>:<hf_filename> to enable download",
                url="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q5_K_M.gguf",
            ),
            "z-image-turbo": ModelSource(
                filename="z_image_turbo-Q6_K.gguf",
                repo_id="",
                notes="set CHIMERA_MODEL_Z_IMAGE_TURBO=<repo_id>:<hf_filename> to enable download",
                url="https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q6_K.gguf",
            ),
            "ae": ModelSource(
                filename="ae.safetensors",
                repo_id="black-forest-labs/FLUX.1-schnell",
                hf_filename="ae.safetensors",
                url="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
            ),
            "bge-small-en": ModelSource(
                filename="bge-small-en-v1.5-q8_0.gguf",
                repo_id="CompendiumLabs/bge-small-en-v1.5-gguf",
                url="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
            ),
            "whisper-base-en": ModelSource(
                filename="ggml-base.en.bin",
                repo_id="ggerganov/whisper.cpp",
                url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            ),
        }

    def apply_env_overrides(self) -> None:
        """Allow overriding repo ids via env vars (CHIMERA_MODEL_<KEY>=repo:file)."""
        for key, src in self.sources.items():
            env_key = "CHIMERA_MODEL_" + key.upper().replace("-", "_")
            val = os.environ.get(env_key)
            if not val:
                continue
            if ":" in val:
                repo, fname = val.split(":", 1)
                src.repo_id = repo
                src.hf_filename = fname
            else:
                src.repo_id = val

    # -- downloads ----------------------------------------------------------

    @staticmethod
    def download_urllib(url: str, dest: Path) -> None:
        print(f"downloading {url} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        last_report = time.monotonic()
        bytes_read = 0
        chunk = 1024 * 1024  # 1 MiB
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total_hdr = r.headers.get("Content-Length")
            total = int(total_hdr) if total_hdr and total_hdr.isdigit() else None
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                bytes_read += len(buf)
                now = time.monotonic()
                if now - last_report >= 2.0:
                    if total:
                        pct = 100.0 * bytes_read / total
                        print(
                            f"  {bytes_read / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"  {bytes_read / 1e6:.1f} MB", flush=True)
                    last_report = now
        tmp.rename(dest)

    @staticmethod
    def download_hf(repo_id: str, filename: str, dest: Path) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print(
                "error: huggingface_hub not installed. Install with: pip install huggingface_hub",
                file=sys.stderr,
            )
            sys.exit(2)
        print(f"downloading {repo_id}:{filename} -> {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Land the file directly in the models dir rather than copying from the
        # HF cache. Newer huggingface_hub uses `local_dir_use_symlinks=False`
        # and places the file at `<local_dir>/<filename>`; older releases
        # fall back to the cache path which we then copy.
        try:
            out = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest.parent),
                local_dir_use_symlinks=False,
            )
        except TypeError:
            # Older huggingface_hub without local_dir kwarg.
            out = hf_hub_download(repo_id=repo_id, filename=filename)
        out_path = Path(out)
        if out_path != dest:
            shutil.copyfile(out_path, dest)

    # -- lookups ------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.paths.models_dir / self.sources[key].filename

    def ensure_model(self, key: str) -> Path:
        src = self.sources[key]
        dest = self.path_for(key)
        if dest.exists():
            return dest
        if src.url:
            self.download_urllib(src.url, dest)
        elif src.repo_id:
            self.download_hf(src.repo_id, src.hub_filename(), dest)
        else:
            raise ModelSourceUnavailable(f"no source configured for model '{key}' ({src.filename}). {src.notes}")
        return dest

    def ensure_models(self, keys: list[str]) -> dict[str, Path]:
        return {k: self.ensure_model(k) for k in keys}

    # -- data assets --------------------------------------------------------
    #
    # These are inputs rather than models. In a checkout that has run
    # `make deps`, jfk.wav already exists under build/whisper.cpp/samples;
    # standalone it does not, so it is fetched from whisper.cpp. The corpus is
    # synthesised rather than downloaded -- the suite has to have something to
    # index even when the checkout's own docs are not there.

    def ensure_corpus(self) -> Path:
        """Path to a line-per-text corpus, preferring the checkout's own."""
        repo_copy = self.paths.find_data_asset("corpus1.txt")
        if repo_copy is not None:
            return repo_copy
        generated = self.paths.models_dir / "corpus_generated.txt"
        if not generated.exists():
            print(f"writing generated corpus -> {generated}")
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("\n".join(self.GENERATED_CORPUS) + "\n", encoding="utf-8")
        return generated

    def ensure_audio(self) -> Path:
        """Path to the jfk.wav sample, downloading it if the checkout lacks one."""
        repo_copy = self.paths.find_data_asset("jfk.wav")
        if repo_copy is not None:
            return repo_copy
        dest = self.paths.models_dir / "jfk.wav"
        if not dest.exists():
            self.download_urllib(self.JFK_WAV_URL, dest)
        return dest

    def ingest_files(self) -> list[Path]:
        """What the rag cases ingest.

        The `scripts/case` originals indexed the checkout's own README.md and
        CHANGELOG.md, which is the more realistic corpus and the one the
        canned query ("how do I offload weights to the CPU?") was written for.
        Standalone those files are not there, so fall back to the generated
        corpus -- which carries a line about CPU offload for exactly this reason.
        """
        docs = [self.paths.root / name for name in ("README.md", "CHANGELOG.md")]
        present = [d for d in docs if d.exists()]
        return present or [self.ensure_corpus()]


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts that were in scripts/case)
# ---------------------------------------------------------------------------

TestFn = Callable[[str, "float | None"], int]


class TestSuite:
    """The smoke-test cases, grouped into families.

    Every case has the same signature -- ``(backend, timeout) -> exit code`` --
    and its docstring is the one-line description `list tests` and the generated
    Makefile print, so keep them short.
    """

    # Every test family, in the order `test-all` runs them: cheap and
    # fast-failing first, the multi-minute image cases last.
    FAMILY_ORDER: tuple[str, ...] = ("embed", "transcribe", "gen", "rag", "sd")

    # What `run --fast` runs in place of `test-all`. The sd cases dominate the
    # wall clock and mostly re-exercise the same three modules, so the fourth --
    # cpu-offload plus flash-attn, the recipe an 8 GiB card actually needs --
    # stands in for all of them. gen-3 is left out rather than the family being
    # named as a whole: `gemma-e4b` has no configured download source, so on a
    # machine without that file already on disk the case is a skip, and a skip
    # is rc=2 -- which would stop the sequence before `clean` over a missing
    # model rather than a bad release.
    FAST_TARGETS: tuple[str, ...] = ("test-embed-1", "test-gen-1", "test-sd-4")

    # Human-readable section headings for the generated Makefile's help text.
    FAMILY_TITLES: dict[str, str] = {
        "embed": "Embedding",
        "transcribe": "Transcription",
        "gen": "Generation",
        "rag": "Vector store / RAG",
        "sd": "Stable Diffusion",
    }

    def __init__(self, env: Env, models: ModelRegistry) -> None:
        self.env = env
        self.models = models
        self.families: dict[str, dict[str, TestFn]] = {
            "embed": {"1": self.embed_1, "2": self.embed_2},
            "transcribe": {"1": self.transcribe_1, "2": self.transcribe_2},
            "gen": {"1": self.gen_1, "2": self.gen_2, "3": self.gen_3},
            "rag": {"1": self.rag_1, "2": self.rag_2},
            "sd": {"1": self.sd_1, "2": self.sd_2, "3": self.sd_3, "4": self.sd_4},
        }
        # Declared separately from FAMILY_ORDER so a family added to one and not
        # the other is caught here rather than silently skipped by `test-all`.
        assert tuple(self.families) == self.FAMILY_ORDER, "families must match FAMILY_ORDER"
        # Same reasoning: a renamed case would otherwise turn `run --fast` into an
        # argparse KeyError deep in the sequence, after the install step has run.
        unknown = [t for t in self.FAST_TARGETS if t not in self.targets()]
        assert not unknown, f"FAST_TARGETS names no such target: {unknown}"

    # -- output bookkeeping -------------------------------------------------

    @property
    def out_dir(self) -> Path:
        """Everything the suite writes goes here, so `clean` sweeps exactly
        what a run produced instead of globbing the project root."""
        self.env.paths.out_dir.mkdir(parents=True, exist_ok=True)
        return self.env.paths.out_dir

    def rag_db(self, name: str) -> Path:
        """Vector store for one rag case.

        Always passed explicitly. Without ``--db`` chimera opens ``$CHIMERA_DB``
        or the platform default -- i.e. the user's real database -- and a smoke
        test has no business creating collections in it.
        """
        return self.out_dir / f"{name}.db"

    # -- stable diffusion ---------------------------------------------------
    #
    # The four cases are the surviving distinct shapes of scripts/case/z_turbo*.sh:
    # bare, flash-attn, cfg-1 + flash-attn, and cfg-1 + offload + flash-attn.
    # Z-Image Turbo is a split-checkpoint model, so all four use the component
    # flags (--diffusion-model / --vae / --llm) rather than -m, and none pass
    # --gpu-layers: `sd` picks up the GPU on its own.

    SD_SIZE: tuple[str, ...] = ("-H", "1024", "-W", "512")
    SD_PROMPT = "a lovely plump cat"

    def sd_case(self, n: str, extra: list[str], timeout: float | None) -> int:
        paths = self.models.ensure_models(ModelRegistry.SD_REQUIREMENTS)
        return self.env.chimera(
            [
                "sd",
                "--diffusion-model",
                str(paths["z-image-turbo"]),
                "--vae",
                str(paths["ae"]),
                "--llm",
                str(paths["qwen3-4b"]),
                *extra,
                *self.SD_SIZE,
                "-o",
                str(self.out_dir / f"z_turbo_{n}.png"),
                "-p",
                self.SD_PROMPT,
            ],
            timeout=timeout,
        )

    def sd_1(self, _backend: str, timeout: float | None) -> int:
        """z_turbo baseline (all on GPU)."""
        # The unqualified run: every module resident on the GPU, ~9.4 GiB of
        # weights. It is here as the control, not because it fits everywhere --
        # on an 8 GiB card this is the case that OOMs, and its failing while
        # sd-4 passes is the useful signal rather than a bug in the release.
        return self.sd_case("1", [], timeout)

    def sd_2(self, _backend: str, timeout: float | None) -> int:
        """z_turbo flash-attn."""
        return self.sd_case("2", ["--diffusion-fa"], timeout)

    def sd_3(self, _backend: str, timeout: float | None) -> int:
        """z_turbo cfg-1 + flash-attn."""
        # Z-Image Turbo is distilled: --cfg-scale 1.0 disables the negative
        # pass, which is both correct for the model and roughly halves the work.
        return self.sd_case("3", ["--cfg-scale", "1.0", "--diffusion-fa"], timeout)

    def sd_4(self, _backend: str, timeout: float | None) -> int:
        """z_turbo cfg-1 + cpu-offload + flash-attn."""
        # The recipe docs/cheatsheet.md gives for Z-Image Turbo, and the one an
        # 8 GiB card needs: weights stream from RAM while compute stays on the
        # GPU. Do not "improve" it with --clip-on-cpu / --vae-on-cpu -- those
        # move the *compute* as well and make the run dramatically slower.
        return self.sd_case("4", ["--cfg-scale", "1.0", "--offload-to-cpu", "--diffusion-fa"], timeout)

    # -- generation ---------------------------------------------------------

    def gen_1(self, backend: str, timeout: float | None) -> int:
        """Llama-3.2-1B short prompt."""
        model = self.models.ensure_model("llama-3.2-1b")
        return self.env.chimera(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "Explain quantum entanglement in one paragraph.",
                "-n",
                "256",
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )

    def gen_2(self, backend: str, timeout: float | None) -> int:
        """Qwen3-4B, same shape as gen-1."""
        # Output streams to stdout as it is produced; there is no --stream flag
        # to ask for it, and no --stats to summarise it afterwards.
        model = self.models.ensure_model("qwen3-4b")
        return self.env.chimera(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "Write a haiku about GPUs.",
                "-n",
                "256",
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )

    def gen_3(self, backend: str, timeout: float | None) -> int:
        """Gemma-4-E4B with sampler knobs."""
        # The temperature flag is --temp (the sd-cli spelling), not --temperature.
        model = self.models.ensure_model("gemma-e4b")
        return self.env.chimera(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "List three interesting facts about octopuses.",
                "-n",
                "512",
                "--temp",
                "0.7",
                "--top-p",
                "0.95",
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )

    # -- embedding ----------------------------------------------------------

    def embed_1(self, backend: str, timeout: float | None) -> int:
        """multi-text vectors via --embd-separator."""
        # `embed` has no --similarity/--threshold: it emits vectors, and ranking
        # a corpus by similarity is what the rag family does. --embd-separator
        # splits one -p into several texts and prints one vector each, which is
        # the closest thing to a batch the subcommand offers.
        model = self.models.ensure_model("bge-small-en")
        return self.env.chimera(
            [
                "embed",
                "-m",
                str(model),
                "-p",
                "death and dying;grief and mourning;a cheerful summer picnic",
                "--embd-separator",
                ";",
                "--embd-output-format",
                "array",
                "--pooling",
                "mean",
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )

    def embed_2(self, backend: str, timeout: float | None) -> int:
        """corpus file -> one vector per line, memoized."""
        model = self.models.ensure_model("bge-small-en")
        corpus = self.models.ensure_corpus()
        out = self.out_dir / "corpus_vectors.txt"
        # --cache-embeddings writes into --cache-db, which defaults to the
        # user's real database; point it at the scratch dir like the rag cases do.
        rc = self.env.chimera(
            [
                "embed",
                "-m",
                str(model),
                "-f",
                str(corpus),
                "--embd-separator",
                "\n",
                "--embd-output-format",
                "raw",
                "-o",
                str(out),
                "--cache-embeddings",
                "--cache-db",
                str(self.rag_db("embed_cache")),
                *self.env.gpu_args(backend),
            ],
            timeout=timeout,
        )
        if rc != 0:
            return rc
        if not out.exists() or out.stat().st_size == 0:
            print(f"error: -o was given but no vectors were written to {out}", file=sys.stderr)
            return 1
        return 0

    # -- transcription ------------------------------------------------------

    def transcribe_1(self, _backend: str, timeout: float | None) -> int:
        """jfk.wav speech-to-text."""
        # The subcommand is `whisper`, not `transcribe`, and the audio flag is
        # -i/--input, not -f. whisper takes no --gpu-layers: it offloads whole
        # encoders, and picks the device itself (--no-gpu / --device opt out).
        model = self.models.ensure_model("whisper-base-en")
        audio = self.models.ensure_audio()
        return self.env.chimera(
            ["whisper", "-m", str(model), "-i", str(audio), "--timestamps"],
            timeout=timeout,
        )

    def transcribe_2(self, _backend: str, timeout: float | None) -> int:
        """jfk.wav -> srt / vtt / json files."""
        model = self.models.ensure_model("whisper-base-en")
        audio = self.models.ensure_audio()
        stem = self.out_dir / "jfk"
        expected = [stem.with_suffix(ext) for ext in (".srt", ".vtt", ".json")]
        for path in expected:
            path.unlink(missing_ok=True)  # so an old run cannot pass this for us
        rc = self.env.chimera(
            [
                "whisper",
                "-m",
                str(model),
                "-i",
                str(audio),
                "--output-file",
                str(stem),
                "--output-srt",
                "--output-vtt",
                "--output-json",
            ],
            timeout=timeout,
        )
        if rc != 0:
            return rc
        missing = [p.name for p in expected if not p.exists() or p.stat().st_size == 0]
        if missing:
            print(f"error: whisper reported success but wrote no {', '.join(missing)}", file=sys.stderr)
            return 1
        return 0

    # -- rag ----------------------------------------------------------------

    def rag_1(self, backend: str, timeout: float | None) -> int:
        """index create + ingest + search via $CHIMERA_DB."""
        # The no---db spelling: chimera falls back to $CHIMERA_DB, so this case
        # covers that path while still keeping its collections out of the user's
        # real database. There is no `chimera rag` subcommand -- retrieval is
        # `index create` -> `index ingest` -> `search`, and the collection
        # records the embedding model, so `search` does not take -e.
        model = self.models.ensure_model("bge-small-en")
        db = self.rag_db("rag_env")
        db.unlink(missing_ok=True)  # start from nothing so the create path is covered
        env = {"CHIMERA_DB": str(db)}
        gpu = self.env.gpu_args(backend)

        rc = self.env.chimera(
            ["index", "create", "-n", "docs", "-e", str(model), *gpu],
            env=env,
            timeout=timeout,
        )
        if rc != 0:
            return rc
        rc = self.env.chimera(
            ["index", "ingest", "-n", "docs", *sum((["-f", str(f)] for f in self.models.ingest_files()), []), *gpu],
            env=env,
            timeout=timeout,
        )
        if rc != 0:
            return rc
        if not db.exists():
            print(f"error: $CHIMERA_DB was set but no store was created at {db}", file=sys.stderr)
            return 1
        return self.env.chimera(
            ["search", "-n", "docs", "-q", "how do I offload weights to the CPU?", "-k", "5", "--mode", "hybrid", *gpu],
            env=env,
            timeout=timeout,
        )

    def rag_2(self, backend: str, timeout: float | None) -> int:
        """explicit --db: ingest, all three retrieval modes, stats, drop."""
        # --db is a per-subcommand flag, not a global one, so every step below
        # repeats it. The three modes are the point of the case: `lexical` never
        # loads the embedding model at all, `semantic` is vec0 KNN, and `hybrid`
        # fuses them -- three different code paths over one store.
        model = self.models.ensure_model("bge-small-en")
        db = self.rag_db("rag_explicit")
        db.unlink(missing_ok=True)
        gpu = self.env.gpu_args(backend)
        dbf = ["--db", str(db)]

        steps: list[list[str]] = [
            ["index", "create", "-n", "docs", "-e", str(model), *dbf, *gpu],
            ["index", "ingest", "-n", "docs", *sum((["-f", str(f)] for f in self.models.ingest_files()), []), *dbf, *gpu],
            ["index", "list", *dbf],
            ["index", "stats", "-n", "docs", *dbf],
            ["search", "-n", "docs", "-q", "how do I offload weights to the CPU?", "-k", "5", "--mode", "semantic", *dbf, *gpu],
            ["search", "-n", "docs", "-q", "Kilimanjaro", "-k", "5", "--mode", "lexical", *dbf],
            ["search", "-n", "docs", "-q", "how do I offload weights to the CPU?", "-k", "5", "--mode", "hybrid", *dbf, *gpu],
            # `db status` runs pending migrations and prints the schema version;
            # cheap, and it is the only thing in the suite that touches the
            # migration path on a store this run just created.
            ["db", "status", *dbf],
            ["index", "drop", "-n", "docs", *dbf],
        ]
        for argv in steps:
            rc = self.env.chimera(argv, timeout=timeout)
            if rc != 0:
                return rc
        return 0

    # -- target bookkeeping -------------------------------------------------

    def targets(self) -> dict[str, tuple[str, str]]:
        """Map each ``test-*`` target name to the (family, case) it runs.

        One token per test -- ``test-all``, ``test-gen-all``, ``test-sd-3`` -- so
        the CLI and the generated Makefile name the same things.
        """
        targets: dict[str, tuple[str, str]] = {"test-all": ("all", "all")}
        for fam, mapping in self.families.items():
            for n in sorted(mapping):
                targets[f"test-{fam}-{n}"] = (fam, n)
            targets[f"test-{fam}-all"] = (fam, "all")
        return targets

    def describe(self, kind: str, n: str) -> str:
        """One-line description of a target, from the case's docstring."""
        if kind == "all":
            return "every test in every family"
        if n == "all":
            return f"all {kind} tests"
        return (self.families[kind][n].__doc__ or "").strip()

    def collect_runs(self, kind: str, n: str) -> list[tuple[str, str]]:
        """Expand ('all'|<family>, 'all'|'1'|...) into concrete (kind, n) pairs."""
        kinds = list(self.families) if kind == "all" else [kind]
        runs: list[tuple[str, str]] = []
        for k in kinds:
            mapping = self.families[k]
            if n == "all":
                runs.extend((k, nk) for nk in sorted(mapping))
            elif n in mapping:
                runs.append((k, n))
            elif kind != "all":
                # An explicit `test embed 3` is a mistake worth reporting; the same
                # number under `test all 3` just means "the families that have a 3".
                print(
                    f"error: no test '{n}' in family '{k}' (have: {', '.join(sorted(mapping))})",
                    file=sys.stderr,
                )
                sys.exit(2)
        if not runs:
            print(f"error: no tests matched kind={kind} n={n}", file=sys.stderr)
            sys.exit(2)
        return runs

    def run_case(self, kind: str, n: str, backend: str, timeout: float | None) -> int:
        return self.families[kind][n](backend, timeout)


# ---------------------------------------------------------------------------
# generated Makefile
# ---------------------------------------------------------------------------


class MakefileRenderer:
    """Renders the Makefile whose rules mirror this script's own targets.

    Written to a *separate* file (`-o rwt.mk`, included from the main Makefile
    if wanted) rather than to ./Makefile: chimera's Makefile is the build
    system, and this one is a frontend for a script that tests binaries the
    build system has already produced.
    """

    PY_VAR = "python3 scripts/rwt.py"

    def __init__(self, env: Env, suite: TestSuite) -> None:
        self.env = env
        self.suite = suite
        self.lines: list[str] = []

    def render(self) -> str:
        self.lines = []
        backends = list(self.env.BACKENDS)

        family_targets: dict[str, list[str]] = {
            fam: [f"test-{fam}-{n}" for n in sorted(mapping)] + [f"test-{fam}-all"]
            for fam, mapping in self.suite.families.items()
        }
        width = max(len(t) for ts in family_targets.values() for t in ts) + 2

        # Group .PHONY into readable lines
        groups = [
            ["help", "info", "clean"],
            [f"install-{b}" for b in backends],
            [f"run-{b}" for b in backends],
            [f"run-{b}-fast" for b in backends],
            ["list-models", "list-tests", "download"],
            *family_targets.values(),
            ["test-all"],
        ]
        phony_lines = " \\\n\t\t".join(" ".join(g) for g in groups if g)

        add = self.lines.append
        add("")
        add(f"PY := {self.PY_VAR}")
        add("")
        add(f".PHONY: {phony_lines}")
        add("")
        add("help:")
        add('\t@echo "Available targets (frontend for $(PY)):"')
        add('\t@echo ""')
        add('\t@echo "  Setup:"')
        add('\t@echo "    info         - show the binary under test and its backends"')
        add('\t@echo "    clean        - remove the installed binary and any test output"')
        for b in backends:
            add(f'\t@echo "    install-{b:<7} - download the latest {b} release into build/rwt/"')
        add('\t@echo ""')
        add('\t@echo "  Models:"')
        add('\t@echo "    list-models  - list known models and whether they are on disk"')
        add('\t@echo "    download     - download all known models (use $(PY) download <key> for one)"')

        for fam, mapping in self.suite.families.items():
            title = self.suite.FAMILY_TITLES.get(fam, fam)
            add('\t@echo ""')
            add(f'\t@echo "  {title} tests (backend auto-detected):"')
            for n in sorted(mapping):
                doc = (mapping[n].__doc__ or "").strip().rstrip(".")
                label = f"test-{fam}-{n}"
                add(f'\t@echo "    {label:<{width}}- {doc}"')
            label = f"test-{fam}-all"
            add(f'\t@echo "    {label:<{width}}- run all {fam} tests"')

        add('\t@echo ""')
        add('\t@echo "  Full cycle (install + test-all + clean):"')
        for b in backends:
            add(f'\t@echo "    run-{b:<8} - install, test and clean the {b} backend"')
        fast = ", ".join(self.suite.FAST_TARGETS)
        add(f'\t@echo "    run-<backend>-fast - as above, but {fast} in place of test-all"')
        add('\t@echo ""')
        add('\t@echo "    list         - list test targets and models"')
        add('\t@echo "    test-all     - run every test in every family"')

        self.rule("info", "info")
        self.rule("clean", "clean")
        for b in backends:
            self.rule(f"install-{b}", f"install --{b}")
        for b in backends:
            self.rule(f"run-{b}", f"run --{b}")
            self.rule(f"run-{b}-fast", f"run --{b} --fast")
        self.rule("list-models", "list models")
        self.rule("list-tests", "list tests")
        self.rule("download", "download all")
        for target in self.suite.targets():
            if target != "test-all":
                self.rule(target, f"test {target}")
        self.rule("test-all", "test test-all")
        add("")
        return "\n".join(self.lines)

    def rule(self, target: str, args: str) -> None:
        self.lines.append("")
        self.lines.append(f"{target}:")
        self.lines.append(f"\t@$(PY) {args}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class Cli:
    """Argparse wiring and the subcommand implementations.

    The parser is built against the *defaults* (so --help can quote them), then
    :meth:`configure` rebuilds the collaborators from what was actually parsed.
    """

    def __init__(self) -> None:
        self.paths = Paths.from_environ()
        self.release = Release()
        self.env = Env(self.paths)
        self.models = ModelRegistry(self.paths)
        self.suite = TestSuite(self.env, self.models)

    # -- configuration ------------------------------------------------------

    def configure(self, args: argparse.Namespace) -> None:
        """Apply parsed options; every collaborator is rebuilt from them."""
        # First, so the directories derived from it move too; the explicit
        # --bin-dir / --out-dir / --data-dir below then override what they name.
        if getattr(args, "build_dir", None):
            self.paths.rebase(Path(args.build_dir).expanduser().resolve())
        if getattr(args, "models_dir", None):
            self.paths.models_dir = Path(args.models_dir).expanduser().resolve()
        if getattr(args, "data_dir", None):
            self.paths.data_dir = Path(args.data_dir).expanduser().resolve()
        if getattr(args, "bin_dir", None):
            self.paths.bin_dir = Path(args.bin_dir).expanduser().resolve()
        if getattr(args, "out_dir", None):
            self.paths.out_dir = Path(args.out_dir).expanduser().resolve()
        if getattr(args, "repo", None):
            self.release.repo = args.repo
        # --bin wins over --bin-dir: it names the executable outright, which is
        # how a build/chimera or a system install is tested without an `install`.
        self.env.bin_path = Path(args.bin).expanduser().resolve() if getattr(args, "bin", None) else None
        self.env.gpu_layers_override = getattr(args, "gpu_layers", None)

    # -- simple commands ----------------------------------------------------

    def cmd_info(self, _args: argparse.Namespace) -> int:
        version, built, loaded = self.env.detect()
        print(f"{'binary:':<10}{self.env.bin_path}{'' if self.env.bin_path.exists() else '  (not installed)'}")
        print(f"{'version:':<10}{version or '(unknown)'}")
        print(f"{'built:':<10}{built or '(unknown)'}")
        print(f"{'loaded:':<10}{loaded or '(unknown)'}")
        print(f"{'models:':<10}{self.paths.models_dir}")
        print(f"{'build:':<10}{self.paths.build_dir}")
        print(f"{'rwt:':<10}{self.paths.rwt_dir}")
        print(f"{'output:':<10}{self.paths.out_dir}")
        print(f"{'host:':<10}{'-'.join(Release.host())}")
        if version:
            print()
            self.env.chimera(["info"])
        return 0

    def cmd_clean(self, _args: argparse.Namespace) -> int:
        # Only ever removes the binary this script installed. An explicit --bin
        # points at something the caller owns -- a build/chimera, a system
        # install -- and deleting that would be a nasty surprise.
        installed = self.paths.bin_dir / self.env.exe_name
        if self.env.bin_override is not None:
            print(f"keeping {self.env.bin_path} (named by --bin, not installed here)")
        elif installed.exists():
            print(f"removing {installed}")
            installed.unlink()
        for path in (self.paths.out_dir, self.paths.cache_dir):
            if path.exists():
                print(f"removing {path}")
                shutil.rmtree(path)
        # All of the above normally live in <build>/rwt, so once they are gone
        # the directory itself is this script's last trace; drop it too. Guarded
        # on emptiness rather than removed outright, since --bin-dir / --out-dir
        # can point elsewhere and something else may own what is left.
        if self.paths.rwt_dir.is_dir() and not any(self.paths.rwt_dir.iterdir()):
            print(f"removing {self.paths.rwt_dir}")
            self.paths.rwt_dir.rmdir()
        return 0

    # -- install ------------------------------------------------------------

    def resolve_asset(self, args: argparse.Namespace) -> tuple[str | None, Path | None]:
        """What ``--asset`` asks to install, as (url, local path).

        The value is a URL, a local archive, or a bare release-asset filename,
        told apart by shape rather than by a second flag: anything with a scheme
        is a URL, anything carrying a path separator or naming a file that
        exists is local, and everything else is treated as an asset name to
        fetch from the release ``--version`` selects.
        """
        value = args.asset
        if not value:
            return None, None

        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return value, None

        path = Path(value).expanduser()
        looks_local = path.exists() or "/" in value or "\\" in value
        if looks_local:
            resolved = path.resolve()
            if not resolved.exists():
                print(f"error: archive not found: {resolved}", file=sys.stderr)
                sys.exit(2)
            return None, resolved

        tag = args.version or self.release.latest_tag()
        return Release.DOWNLOAD.format(repo=self.release.repo, tag=tag, asset=value), None

    def cmd_install(self, args: argparse.Namespace) -> int:
        target = self.paths.bin_dir / self.env.exe_name
        if target.exists() and not args.force:
            # Re-installing over a binary the caller may be mid-investigation on
            # is the kind of thing worth asking for explicitly.
            print(f"{target} already exists; pass --force to replace it")
            return 0

        try:
            url, local = self.resolve_asset(args)
            if url is None and local is None:
                backend = getattr(args, "backend", None)
                if not backend:
                    flags = "/".join("--" + b for b in self.env.BACKENDS)
                    print(f"error: give a backend ({flags}), or --asset <path-or-url>", file=sys.stderr)
                    return 2
                tag = args.version or self.release.latest_tag()
                # The workflows strip a leading `v` from the tag for the
                # filename, so the asset is named for the version, not the tag.
                url = self.release.url_for(backend, tag.lstrip("v"), tag=tag)

            if local is None:
                assert url is not None
                local = Release.download(url, self.paths.cache_dir / Path(url).name)
            Release.extract(local, self.paths.bin_dir)
        except AssetUnavailable as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except urllib.error.HTTPError as e:
            print(f"error: {e.code} fetching {e.url}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError, tarfile.TarError, zipfile.BadZipFile) as e:
            print(f"error: install failed: {e}", file=sys.stderr)
            return 2

        # Report what actually landed, since the asset name is the only thing
        # that has been checked so far and it is chosen, not verified.
        version, built, loaded = self.env.detect()
        print(f"chimera {version or '?'} (built: {built or '?'}, loaded: {loaded or '?'}) at {self.env.bin_path}")
        return 0

    # -- registries ---------------------------------------------------------

    def cmd_download(self, args: argparse.Namespace) -> int:
        keys = list(self.models.sources) if args.key == "all" else [args.key]
        failures = 0
        for k in keys:
            try:
                path = self.models.ensure_model(k)
                print(f"ok: {k} -> {path}")
            except ModelSourceUnavailable as e:
                print(f"skip: {k}: {e}", file=sys.stderr)
                failures += 1
        return 1 if failures else 0

    def cmd_list_models(self, _args: argparse.Namespace) -> int:
        for key, src in self.models.sources.items():
            source = f"hf:{src.repo_id}:{src.hub_filename()}" if src.repo_id else (src.url or "(no source configured)")
            on_disk = "YES" if (self.paths.models_dir / src.filename).exists() else "no"
            print(f"{key:<16} file={src.filename:<40} on_disk={on_disk:<3} source={source}")
            if src.notes and not src.repo_id and not src.url:
                print(f"{'':<16} note: {src.notes}")
        return 0

    def cmd_list_tests(self, _args: argparse.Namespace) -> int:
        targets = self.suite.targets()
        width = max(len(t) for t in targets)
        for target, (kind, n) in targets.items():
            print(f"{target:<{width}}  {self.suite.describe(kind, n)}")
        return 0

    def cmd_list_assets(self, _args: argparse.Namespace) -> int:
        """The release assets this script knows how to install, per backend."""
        system, arch = Release.host()
        print(f"host: {system}-{arch}")
        for backend in self.env.BACKENDS:
            try:
                name = Release.asset_name(backend, "<version>")
            except AssetUnavailable as e:
                print(f"{backend:<8} -- {e}")
                continue
            print(f"{backend:<8} {name}")
        return 0

    def cmd_list(self, args: argparse.Namespace) -> int:
        """`list` with no argument shows every registry; `list tests|models|assets` narrows."""
        what = getattr(args, "what", "all")
        rc = 0
        sections: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
            ("tests", self.cmd_list_tests),
            ("models", self.cmd_list_models),
            ("assets", self.cmd_list_assets),
        ]
        for name, fn in sections:
            if what in (name, "all"):
                if what == "all":
                    print(f"{name}:" if name == "tests" else f"\n{name}:")
                rc |= fn(args)
        return rc

    def cmd_gen_makefile(self, args: argparse.Namespace) -> int:
        content = MakefileRenderer(self.env, self.suite).render()
        if args.output:
            Path(args.output).write_text(content)
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(content)
        return 0

    # -- test ---------------------------------------------------------------

    @staticmethod
    def _use_color(no_color: bool) -> bool:
        if no_color or os.environ.get("NO_COLOR"):
            return False
        return sys.stdout.isatty()

    def cmd_test(self, args: argparse.Namespace) -> int:
        kind, n = self.suite.targets()[args.target]

        # --dry-run promises to touch nothing, so it precedes every other step.
        if args.dry_run:
            backend = getattr(args, "backend", None) or self.env.detect_backend() or "?"
            for k, case in self.suite.collect_runs(kind, n):
                print(f"would run: {k} {case} (backend={backend})")
            return 0

        backend = self.env.require_backend(getattr(args, "backend", None))
        runs = self.suite.collect_runs(kind, n)

        problem = self.env.preflight(backend)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

        color = self._use_color(args.no_color)
        green = "\033[32m" if color else ""
        red = "\033[31m" if color else ""
        reset = "\033[0m" if color else ""

        results: list[tuple[str, str, int, float]] = []
        for k, case in runs:
            print(f"\n=== {k} test {case} (backend={backend}) ===")
            started = time.monotonic()
            try:
                rc = self.suite.run_case(k, case, backend, args.timeout)
            except ModelSourceUnavailable as e:
                print(f"skip: {e}", file=sys.stderr)
                rc = 2
            results.append((k, case, rc, time.monotonic() - started))
            if rc != 0 and args.fail_fast:
                break

        # Summary
        print("\n=== summary ===")
        worst = 0
        for k, case, rc, secs in results:
            status = f"{green}PASS{reset}" if rc == 0 else f"{red}FAIL (rc={rc}){reset}"
            print(f"  {k} {case}: {status}  ({secs:.1f}s)")
            worst = max(worst, rc)
        passed = sum(1 for r in results if r[2] == 0)
        total = sum(r[3] for r in results)
        print(f"{passed}/{len(results)} passed in {total:.1f}s")
        return worst

    # -- run ----------------------------------------------------------------

    def run_targets(self, args: argparse.Namespace) -> list[str]:
        """The test targets one `run` invocation covers, in order."""
        if not args.fast:
            return [args.target or "test-all"]
        if args.target is not None:
            print(
                f"error: --fast already names its targets ({', '.join(self.suite.FAST_TARGETS)});"
                f" drop it to run '{args.target}' alone",
                file=sys.stderr,
            )
            sys.exit(2)
        return list(self.suite.FAST_TARGETS)

    def cmd_run(self, args: argparse.Namespace) -> int:
        """install -> test... -> clean, stopping at the first step that fails.

        A failure leaves the binary in place rather than cleaning up after it:
        the thing worth inspecting when a release fails is the binary it failed
        with, and `clean` is one command away once it has been looked at.
        """

        def test_step(target: str) -> Callable[[argparse.Namespace], int]:
            def step(a: argparse.Namespace) -> int:
                a.target = target
                return self.cmd_test(a)

            return step

        targets = self.run_targets(args)
        steps: list[tuple[str, Callable[[argparse.Namespace], int]]] = [
            ("install", self.cmd_install),
            *((f"test {t}", test_step(t)) for t in targets),
            ("clean", self.cmd_clean),
        ]

        if args.dry_run:
            # `test --dry-run` promises to touch nothing, and `run` inherits that
            # promise for the whole sequence: print the steps, run none of them.
            where = f" --bin {self.env.bin_path}" if self.env.bin_override is not None else ""
            for name, _ in steps:
                verb, _, target = name.partition(" ")
                print(f"would run: {SCRIPT_NAME} {verb}{where}{' ' + target if target else ''}")
            print()
            for _, step in steps[1 : 1 + len(targets)]:
                step(args)
            return 0

        for i, (name, step) in enumerate(steps):
            print(f"\n=== {name} ===")
            rc = step(args)
            if rc != 0:
                skipped = ", ".join(n for n, _ in steps[i + 1 :])
                print(f"\nerror: {name} failed (rc={rc}); skipping {skipped}", file=sys.stderr)
                return rc
        return 0

    # -- argparse -----------------------------------------------------------

    def common_parser(self) -> argparse.ArgumentParser:
        """Options accepted both before and after the subcommand."""
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument(
            "--bin",
            metavar="PATH",
            default=argparse.SUPPRESS,
            help="chimera executable to test, instead of the one `install` puts in "
            f"{self.paths.bin_dir}. Use it to point the suite at build/chimera or a "
            "system install; `clean` never deletes a binary named this way.",
        )
        c.add_argument(
            "--build-dir",
            "--build_dir",
            metavar="PATH",
            dest="build_dir",
            default=argparse.SUPPRESS,
            help=f"build tree this script works inside; it puts everything it creates -- the "
            f"installed binary, downloaded archives, test output -- in <build-dir>/rwt "
            f"(default: {self.paths.build_dir}, following the Makefile's BUILD_DIR). Models are "
            "not under it and are not affected.",
        )
        c.add_argument(
            "--bin-dir",
            "--bin_dir",
            metavar="PATH",
            dest="bin_dir",
            default=argparse.SUPPRESS,
            help=f"directory `install` unpacks the release binary into (default: {self.paths.bin_dir})",
        )
        c.add_argument(
            "--models-dir",
            "--models_dir",
            metavar="PATH",
            dest="models_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding the GGUF/safetensors models (default: {self.paths.models_dir})",
        )
        shorthand = c.add_mutually_exclusive_group()
        for backend in self.env.BACKENDS:
            shorthand.add_argument(
                f"--{backend}",
                dest="backend",
                action="store_const",
                const=backend,
                default=argparse.SUPPRESS,
                help=f"the {backend} backend: selects that release asset for `install`, "
                "and asserts it for `test`",
            )
        c.add_argument(
            "--data-dir",
            "--data_dir",
            metavar="PATH",
            dest="data_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding jfk.wav / corpus1.txt (default: {self.paths.data_dir})",
        )
        c.add_argument(
            "--out-dir",
            "--out_dir",
            metavar="PATH",
            dest="out_dir",
            default=argparse.SUPPRESS,
            help=f"where images, transcripts and scratch DBs are written (default: {self.paths.out_dir}); "
            "`clean` removes it wholesale",
        )
        c.add_argument(
            "--repo",
            metavar="OWNER/NAME",
            default=argparse.SUPPRESS,
            help=f"GitHub repository releases are fetched from (default: {self.release.repo})",
        )
        return c

    @staticmethod
    def install_parser() -> argparse.ArgumentParser:
        """Options that only mean something while writing to build/rwt/."""
        i = argparse.ArgumentParser(add_help=False)
        i.add_argument(
            "--version",
            metavar="TAG",
            default=None,
            help="release to install (e.g. 0.2.16). Default: whatever /releases/latest resolves to.",
        )
        i.add_argument(
            "--asset",
            metavar="PATH|URL|NAME",
            default=None,
            help="override what to install: a local archive "
            "(dist/chimera-0.2.16-linux-x86_64-cuda.tar.gz), a full URL, or a bare asset "
            "name to fetch from the --version release. Usually unnecessary -- without it "
            "the backend and the host pick the asset (--cuda on Linux -> "
            "chimera-<version>-linux-x86_64-cuda.tar.gz).",
        )
        i.add_argument(
            "--force",
            action="store_true",
            help="replace an existing binary in --bin-dir instead of leaving it alone",
        )
        return i

    def test_parser(self) -> argparse.ArgumentParser:
        """Options that shape a test run; shared by `test` and `run`."""
        t = argparse.ArgumentParser(add_help=False)
        t.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="per-test timeout in seconds (default: no timeout)",
        )
        t.add_argument(
            "--gpu-layers",
            "--gpu_layers",
            dest="gpu_layers",
            type=int,
            default=None,
            help="layers the llama-side cases offload (default: "
            f"{Env.GPU_LAYERS_DEFAULT} on a GPU backend, 0 on cpu). chimera itself defaults "
            "to 0, so a GPU release tested without this would only measure the CPU path.",
        )
        t.add_argument(
            "--fail-fast",
            action="store_true",
            help="stop at the first failing test instead of running the full matrix",
        )
        t.add_argument(
            "--dry-run",
            action="store_true",
            help="print the test matrix without downloading or invoking anything",
        )
        t.add_argument(
            "--no-color",
            action="store_true",
            help="disable colored PASS/FAIL output in the summary",
        )
        return t

    def build_parser(self) -> argparse.ArgumentParser:
        common = self.common_parser()
        p = argparse.ArgumentParser(
            description="chimera release tester",
            parents=[common],
            epilog="example: rwt.py install --cuda && rwt.py test --cuda test-all --models-dir models",
        )
        _sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

        class sub:  # noqa: N801 - thin shim so add_parser always inherits `common`
            @staticmethod
            def add_parser(
                name: str,
                parents: Sequence[argparse.ArgumentParser] = (),
                **kw: Any,
            ) -> argparse.ArgumentParser:
                return _sub.add_parser(name, parents=[common, *parents], **kw)

        sub.add_parser("info", help="show the binary under test, its backends and the models dir").set_defaults(
            func=self.cmd_info
        )
        sub.add_parser("clean", help="remove the installed binary and everything the suite wrote").set_defaults(
            func=self.cmd_clean
        )

        inst = sub.add_parser(
            "install",
            parents=[self.install_parser()],
            help="download a chimera release and unpack it into --bin-dir",
        )
        inst.set_defaults(func=self.cmd_install)

        dl = sub.add_parser("download", help="download a model (or 'all')")
        dl.add_argument("key", choices=[*self.models.sources.keys(), "all"])
        dl.set_defaults(func=self.cmd_download)

        lst = sub.add_parser("list", help="list test targets, models and release assets (or one of them)")
        lst.add_argument(
            "what",
            nargs="?",
            choices=["tests", "models", "assets", "all"],
            default="all",
            help="which registry to show (default: all of them)",
        )
        lst.set_defaults(func=self.cmd_list)

        # The flat names, kept working but out of --help so `list` is the one
        # obvious spelling.
        sub.add_parser("list-models").set_defaults(func=self.cmd_list_models)
        sub.add_parser("list-tests").set_defaults(func=self.cmd_list_tests)

        gm = sub.add_parser("gen-makefile", help="generate a Makefile from this script's registries")
        gm.add_argument("-o", "--output", help="write to file instead of stdout (e.g. -o rwt.mk)")
        gm.set_defaults(func=self.cmd_gen_makefile)

        # `test` takes one target name -- `test-sd-4` rather than `test sd 4`, so a
        # target is a single token and matches the Makefile rule of the same name.
        t = sub.add_parser("test", parents=[self.test_parser()], help="run a test target (see `list tests`)")
        t.add_argument(
            "target",
            choices=list(self.suite.targets()),
            metavar="TARGET",
            help="one of the targets `list tests` prints, e.g. test-all, test-gen-1",
        )
        t.set_defaults(func=self.cmd_test)

        # `run` is the whole cycle in one command, so a release can be checked on
        # a clean machine without three invocations that must agree on the backend.
        r = sub.add_parser(
            "run",
            parents=[self.install_parser(), self.test_parser()],
            help="install, test, then clean -- stopping at the first failure",
        )
        r.add_argument(
            "target",
            nargs="?",
            default=None,
            choices=list(self.suite.targets()),
            metavar="[TARGET]",
            help="the target to run (default: test-all)",
        )
        r.add_argument(
            "--fast",
            action="store_true",
            help="the short cycle: run "
            + ", ".join(self.suite.FAST_TARGETS)
            + " in place of test-all, skipping the image cases that dominate the wall clock",
        )
        r.set_defaults(func=self.cmd_run)

        return p

    def main(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        self.configure(args)
        return int(args.func(args) or 0)


def main() -> None:
    sys.exit(Cli().main())


if __name__ == "__main__":
    main()

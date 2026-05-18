#!/usr/bin/env python3
"""combine_archives.py: bundle chimera + all transitive static deps into
one fat static library.

Assumes src/chimera/CMakeLists.txt has been converted from add_executable
to add_library(chimera_core STATIC ...), producing libchimera_core.a (or
chimera_core.lib on Windows) somewhere under build/.

Per-platform tool:
    macOS   - libtool -static       (native, handles dup symbols quietly)
    Linux   - GNU ar via MRI script (extracts members, repacks one .a)
    Windows - lib.exe /OUT:...      (MSVC archiver, concatenates .libs)

CRITICAL: three sibling ggml builds exist (one each from llama.cpp,
whisper.cpp, stable-diffusion.cpp). chimera only links the llama.cpp
ggml build (see top-level CMakeLists.txt around line 306). Bundling
more than one ggml will create duplicate symbols. This script only
references the llama.cpp ggml archives.

NOTE: a "bundled" static library does NOT relieve consumers of the
need to pass platform link flags. ggml's GPU backends self-register
via static initializers in object files; without -force_load (macOS),
--whole-archive (GNU ld), or /WHOLEARCHIVE (link.exe), the backend
.o members will be dropped and the backend silently disappears at
runtime. The bundle just shrinks the archive list from ~15 to 1; it
does not change the linker contract.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_ROOT = REPO_ROOT / "build"


@dataclass
class Inventory:
    build_root: Path
    with_whisper: bool = True
    with_sd: bool = True
    with_linenoise: bool = True
    with_metal: bool = True
    with_blas: bool = True
    with_cuda: bool = False
    with_vulkan: bool = False
    with_hip: bool = False
    with_sycl: bool = False
    with_opencl: bool = False
    archive_ext: str = ".a"  # ".lib" on Windows

    archives: list[Path] = field(default_factory=list)

    def collect(self) -> list[Path]:
        e = self.archive_ext
        llama = self.build_root / "llama.cpp" / "build"
        whisper = self.build_root / "whisper.cpp" / "build"
        sd = self.build_root / "stable-diffusion.cpp" / "build"
        linenoise = self.build_root / "linenoise" / "build"

        # chimera_core first - it has unresolved refs that pull
        # members from everything below it. (On GNU ld order matters;
        # on macOS / MSVC the linker iterates so it is cosmetic, but
        # we keep one canonical order for diff-ability.)
        out: list[Path] = [self.build_root / f"libchimera_core{e}"]

        # llama.cpp + server frontend + httplib
        out += [
            llama / "src" / f"libllama{e}",
            llama / "common" / f"libllama-common{e}",
            llama / "common" / f"libllama-common-base{e}",
            llama / "tools" / "mtmd" / f"libmtmd{e}",
            llama / "tools" / "server" / f"libserver-context{e}",
            llama / "vendor" / "cpp-httplib" / f"libcpp-httplib{e}",
        ]

        if self.with_whisper:
            out += [
                whisper / "src" / f"libwhisper{e}",
                whisper / "examples" / f"libcommon{e}",
            ]
        if self.with_sd:
            out += [
                sd / f"libstable-diffusion{e}",
                sd / "thirdparty" / "libwebp" / f"libwebp{e}",
                sd / "thirdparty" / "libwebp" / f"libsharpyuv{e}",
                sd / "thirdparty" / "libwebp" / f"libwebpmux{e}",
                sd / "thirdparty" / "libwebm" / f"libwebm{e}",
            ]

        # ggml LAST - it satisfies everyone else's refs. llama.cpp
        # build only.
        g = llama / "ggml" / "src"
        out += [
            g / f"libggml{e}",
            g / f"libggml-base{e}",
            g / f"libggml-cpu{e}",
        ]
        if self.with_metal:
            out.append(g / "ggml-metal" / f"libggml-metal{e}")
        if self.with_blas:
            out.append(g / "ggml-blas" / f"libggml-blas{e}")
        if self.with_cuda:
            out.append(g / "ggml-cuda" / f"libggml-cuda{e}")
        if self.with_vulkan:
            out.append(g / "ggml-vulkan" / f"libggml-vulkan{e}")
        if self.with_hip:
            out.append(g / "ggml-hip" / f"libggml-hip{e}")
        if self.with_sycl:
            out.append(g / "ggml-sycl" / f"libggml-sycl{e}")
        if self.with_opencl:
            out.append(g / "ggml-opencl" / f"libggml-opencl{e}")

        if self.with_linenoise:
            out.append(linenoise / f"liblinenoise{e}")

        self.archives = out
        return out

    def check(self) -> None:
        missing = [p for p in self.archives if not p.is_file()]
        if not missing:
            return
        print("Missing archives:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nBuild chimera first and ensure libchimera_core (or chimera_core.lib)\n"
            "has been produced. That requires converting src/chimera/CMakeLists.txt\n"
            "from add_executable to add_library(chimera_core STATIC ...).",
            file=sys.stderr,
        )
        sys.exit(1)


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, **kw)


def combine_macos(inv: Inventory, output: Path) -> None:
    """libtool -static merges archive members natively. Duplicate
    symbol names produce a warning, not an error - acceptable here
    because we already filtered the duplicate ggml builds upstream."""
    libtool = shutil.which("libtool")
    if libtool is None:
        sys.exit("libtool not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    run([libtool, "-static", "-o", str(output), *[str(p) for p in inv.archives]])


def combine_linux(inv: Inventory, output: Path) -> None:
    """GNU ar has two ways to merge: an MRI script (ADDLIB) or extract
    and repack. MRI is simpler but several distros ship a broken thin-
    archive interaction. The extract-and-repack path below is bullet-
    proof: extract each input into its own subdir (prevents .o name
    collisions across archives, which silently overwrite otherwise),
    then `ar crs` the union into the output."""
    ar = shutil.which("ar")
    ranlib = shutil.which("ranlib")
    if ar is None or ranlib is None:
        sys.exit("ar/ranlib not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with tempfile.TemporaryDirectory(prefix="chimera-combine-") as tmp:
        tmp_path = Path(tmp)
        all_objs: list[Path] = []
        for i, archive in enumerate(inv.archives):
            sub = tmp_path / f"{i:02d}_{archive.stem}"
            sub.mkdir()
            # `ar x` extracts into cwd; run with cwd=sub for isolation.
            run([ar, "x", str(archive.resolve())], cwd=sub)
            objs = sorted(sub.glob("*.o"))
            if not objs:
                print(f"warning: no .o members in {archive}", file=sys.stderr)
            all_objs.extend(objs)
        # Rename to globally-unique basenames so a single `ar crs` is
        # safe even if two archives shipped identically-named .o files.
        flattened: list[Path] = []
        flat_dir = tmp_path / "flat"
        flat_dir.mkdir()
        for i, obj in enumerate(all_objs):
            dest = flat_dir / f"{i:05d}_{obj.name}"
            shutil.copy2(obj, dest)
            flattened.append(dest)
        # `ar crs` in batches to avoid ARG_MAX on large dep sets.
        BATCH = 500
        for start in range(0, len(flattened), BATCH):
            chunk = flattened[start : start + BATCH]
            mode = "crs" if start == 0 else "rs"
            run([ar, mode, str(output), *[str(o) for o in chunk]])
        run([ranlib, str(output)])


def combine_windows(inv: Inventory, output: Path) -> None:
    """lib.exe accepts multiple input .libs and writes a merged .lib.
    Requires running from an MSVC developer prompt (so lib.exe is on
    PATH and links against the right toolchain)."""
    lib_exe = shutil.which("lib.exe") or shutil.which("lib")
    if lib_exe is None:
        sys.exit(
            "lib.exe not found - run this from an x64 Native Tools Command "
            "Prompt for VS, or set the appropriate vcvars."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # /NOLOGO keeps the output diffable; /OUT: takes the target.
    run(
        [
            lib_exe,
            "/NOLOGO",
            f"/OUT:{output}",
            *[str(p) for p in inv.archives],
        ]
    )


def main() -> int:
    is_windows = platform.system() == "Windows"
    is_macos = platform.system() == "Darwin"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="output archive path (default: build/libchimera_bundle.{a,lib})")
    ap.add_argument("--platform", choices=("auto", "macos", "linux", "windows"), default="auto")
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--no-sd", action="store_true")
    ap.add_argument("--no-linenoise", action="store_true")
    ap.add_argument("--no-metal", action="store_true")
    ap.add_argument("--no-blas", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--vulkan", action="store_true")
    ap.add_argument("--hip", action="store_true")
    ap.add_argument("--sycl", action="store_true")
    ap.add_argument("--opencl", action="store_true")
    ap.add_argument("--list", action="store_true",
                    help="print the resolved archive list and exit")
    args = ap.parse_args()

    target = args.platform
    if target == "auto":
        target = "windows" if is_windows else "macos" if is_macos else "linux"

    ext = ".lib" if target == "windows" else ".a"
    inv = Inventory(
        build_root=args.build_root.resolve(),
        with_whisper=not args.no_whisper,
        with_sd=not args.no_sd,
        with_linenoise=not args.no_linenoise,
        with_metal=(not args.no_metal) and target == "macos",
        with_blas=not args.no_blas,
        with_cuda=args.cuda,
        with_vulkan=args.vulkan,
        with_hip=args.hip,
        with_sycl=args.sycl,
        with_opencl=args.opencl,
        archive_ext=ext,
    )
    inv.collect()

    if args.list:
        for p in inv.archives:
            marker = " " if p.is_file() else "!"
            print(f"{marker} {p}")
        return 0

    inv.check()

    output = args.output
    if output is None:
        name = "chimera_bundle.lib" if target == "windows" else "libchimera_bundle.a"
        output = args.build_root / name

    if target == "macos":
        combine_macos(inv, output)
    elif target == "linux":
        combine_linux(inv, output)
    elif target == "windows":
        combine_windows(inv, output)
    else:
        sys.exit(f"unknown platform: {target}")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"\nWrote {output} ({size_mb:.1f} MB)")
    print("\nConsumer link flags (still required - bundle does not change linker contract):")
    if target == "macos":
        print(f"  -Wl,-force_load,{output}    # or omit force_load and lose GPU backends")
    elif target == "linux":
        print(f"  -Wl,--whole-archive {output} -Wl,--no-whole-archive")
    else:
        print(f"  /WHOLEARCHIVE:{output.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

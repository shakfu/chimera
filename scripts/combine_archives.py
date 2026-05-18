#!/usr/bin/env python3
"""combine_archives.py: bundle chimera's transitive C++ dependencies
into two static libraries.

  libchimera_ggml.a        ggml core + per-backend ggml archives. Must
                           be whole-archived by the consumer (-force_load
                           on macOS, --whole-archive on Linux,
                           /WHOLEARCHIVE on Windows) so the backend
                           static initializers actually run.

  libchimera_thirdparty.a  everything else: llama, llama-common, mtmd,
                           server-context, cpp-httplib, whisper, sd,
                           vendored libwebp / libwebm, linenoise.
                           Linked normally so unused members get pruned.

The split mirrors the contract in src/chimera/CMakeLists.txt: that
build force_loads ggml archives only, and lets the linker prune the
rest. A single merged deps archive would force-load both copies of
helpers like trim() that exist in two upstream codebases (server-
context's util.cpp and whisper-common's common.cpp), producing
duplicate-symbol link failures. Splitting along the whole-archive
boundary is what makes the bundling honest.

libchimera.a is NOT included in either output - it is produced by
`make build` and the consumer links it separately. The full consumer
link line is:

  libchimera.a libchimera_thirdparty.a -<whole-archive> libchimera_ggml.a
"""

from __future__ import annotations

import argparse
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

    # Two groups, populated by collect(). The split corresponds to
    # whether the consumer must whole-archive the group (ggml) or
    # normal-link it (thirdparty).
    ggml: list[Path] = field(default_factory=list)
    thirdparty: list[Path] = field(default_factory=list)

    def collect(self) -> None:
        e = self.archive_ext
        llama = self.build_root / "llama.cpp" / "build"
        whisper = self.build_root / "whisper.cpp" / "build"
        sd = self.build_root / "stable-diffusion.cpp" / "build"
        linenoise = self.build_root / "linenoise" / "build"

        # --- thirdparty: normal-link archives -------------------------
        tp: list[Path] = [
            llama / "src" / f"libllama{e}",
            llama / "common" / f"libllama-common{e}",
            llama / "common" / f"libllama-common-base{e}",
            llama / "tools" / "mtmd" / f"libmtmd{e}",
            llama / "tools" / "server" / f"libserver-context{e}",
            llama / "vendor" / "cpp-httplib" / f"libcpp-httplib{e}",
        ]
        if self.with_whisper:
            tp += [
                whisper / "src" / f"libwhisper{e}",
                whisper / "examples" / f"libcommon{e}",
            ]
        if self.with_sd:
            tp += [
                sd / f"libstable-diffusion{e}",
                sd / "thirdparty" / "libwebp" / f"libwebp{e}",
                sd / "thirdparty" / "libwebp" / f"libsharpyuv{e}",
                sd / "thirdparty" / "libwebp" / f"libwebpmux{e}",
                sd / "thirdparty" / "libwebm" / f"libwebm{e}",
            ]
        if self.with_linenoise:
            tp.append(linenoise / f"liblinenoise{e}")

        # --- ggml: whole-archive group --------------------------------
        # llama.cpp ggml build ONLY. The whisper and sd subprojects
        # produce sibling ggml builds we MUST NOT bundle (would
        # duplicate every ggml symbol); see assert_single_ggml.
        g = llama / "ggml" / "src"
        gg: list[Path] = [
            g / f"libggml{e}",
            g / f"libggml-base{e}",
            g / f"libggml-cpu{e}",
        ]
        if self.with_metal:
            gg.append(g / "ggml-metal" / f"libggml-metal{e}")
        if self.with_blas:
            gg.append(g / "ggml-blas" / f"libggml-blas{e}")
        if self.with_cuda:
            gg.append(g / "ggml-cuda" / f"libggml-cuda{e}")
        if self.with_vulkan:
            gg.append(g / "ggml-vulkan" / f"libggml-vulkan{e}")
        if self.with_hip:
            gg.append(g / "ggml-hip" / f"libggml-hip{e}")
        if self.with_sycl:
            gg.append(g / "ggml-sycl" / f"libggml-sycl{e}")
        if self.with_opencl:
            gg.append(g / "ggml-opencl" / f"libggml-opencl{e}")

        self.thirdparty = tp
        self.ggml = gg

    def all_archives(self) -> list[Path]:
        return self.thirdparty + self.ggml

    def assert_single_ggml(self) -> None:
        """Guard against the duplicate-ggml footgun. Three sibling ggml
        builds exist on disk (llama.cpp, whisper.cpp, stable-diffusion.cpp);
        only llama.cpp's may be bundled. Fails loudly if a future edit
        accidentally pulls in another."""
        forbidden_roots = (
            self.build_root / "whisper.cpp" / "build" / "ggml",
            self.build_root / "stable-diffusion.cpp" / "build" / "ggml",
        )
        offenders: list[Path] = []
        for p in self.all_archives():
            rp = p.resolve()
            for root in forbidden_roots:
                try:
                    rp.relative_to(root.resolve())
                except ValueError:
                    continue
                offenders.append(p)
                break
        if offenders:
            print(
                "FATAL: archive list includes ggml builds from whisper.cpp or\n"
                "stable-diffusion.cpp. Only the llama.cpp ggml build may be\n"
                "bundled; mixing them produces duplicate symbols.\n",
                file=sys.stderr,
            )
            for p in offenders:
                print(f"  forbidden: {p}", file=sys.stderr)
            sys.exit(2)

    def check(self) -> None:
        self.assert_single_ggml()
        missing = [p for p in self.all_archives() if not p.is_file()]
        if not missing:
            return
        print("Missing archives:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        print(
            "\nRun `make build` first to produce the transitive dep archives.",
            file=sys.stderr,
        )
        sys.exit(1)


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, **kw)


def combine_macos(inputs: list[Path], output: Path) -> None:
    """libtool -static merges archive members natively. Duplicate
    symbol names produce a warning, not an error - tolerable inside
    a group, but the ggml/thirdparty split is what keeps the consumer
    link line itself free of duplicate-symbol failures."""
    libtool = shutil.which("libtool")
    if libtool is None:
        sys.exit("libtool not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    run([libtool, "-static", "-o", str(output), *[str(p) for p in inputs]])


def combine_linux(inputs: list[Path], output: Path) -> None:
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
        for i, archive in enumerate(inputs):
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


def combine_windows(inputs: list[Path], output: Path) -> None:
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
    run([lib_exe, "/NOLOGO", f"/OUT:{output}", *[str(p) for p in inputs]])


def combine_for(target: str, inputs: list[Path], output: Path) -> None:
    if target == "macos":
        combine_macos(inputs, output)
    elif target == "linux":
        combine_linux(inputs, output)
    elif target == "windows":
        combine_windows(inputs, output)
    else:
        sys.exit(f"unknown platform: {target}")


def main() -> int:
    is_windows = platform.system() == "Windows"
    is_macos = platform.system() == "Darwin"

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-root", type=Path, default=DEFAULT_BUILD_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="dir to write the two archives (default: build/)")
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
                    help="print the resolved archive groups and exit")
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
        print("# ggml group (consumer must whole-archive):")
        for p in inv.ggml:
            print(f"{' ' if p.is_file() else '!'} {p}")
        print("\n# thirdparty group (normal link):")
        for p in inv.thirdparty:
            print(f"{' ' if p.is_file() else '!'} {p}")
        return 0

    inv.check()

    out_dir = (args.output_dir or args.build_root).resolve()
    if target == "windows":
        ggml_out = out_dir / "chimera_ggml.lib"
        tp_out = out_dir / "chimera_thirdparty.lib"
        chimera_a = args.build_root / "chimera.lib"
    else:
        ggml_out = out_dir / "libchimera_ggml.a"
        tp_out = out_dir / "libchimera_thirdparty.a"
        chimera_a = args.build_root / "libchimera.a"

    combine_for(target, inv.thirdparty, tp_out)
    combine_for(target, inv.ggml, ggml_out)

    tp_mb = tp_out.stat().st_size / (1024 * 1024)
    gg_mb = ggml_out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {tp_out}   ({tp_mb:.1f} MB)")
    print(f"Wrote {ggml_out}  ({gg_mb:.1f} MB)")

    print("\nConsumer link line (whole-archive the ggml group only;")
    print("the thirdparty group must be normal-linked or you will hit")
    print("duplicate-symbol errors on helpers that exist in two upstream TUs):")
    if target == "macos":
        print(f"  {chimera_a} {tp_out} -Wl,-force_load,{ggml_out}")
    elif target == "linux":
        print(f"  {chimera_a} {tp_out} -Wl,--whole-archive {ggml_out} -Wl,--no-whole-archive")
    else:
        print(f"  {chimera_a.name} {tp_out.name} /WHOLEARCHIVE:{ggml_out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

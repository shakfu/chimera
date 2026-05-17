#!/usr/bin/env python3

"""manage.py: chimera build manager.

Drives the third-party dependency build for chimera (llama.cpp, whisper.cpp,
stable-diffusion.cpp). Uses only the Python stdlib. The four-version block at
module scope is parsed by the top-level CMakeLists.txt -- keep it intact.

usage: manage.py [-h] [-v]  ...

chimera build manager

    build        fetch + build llama.cpp / whisper.cpp / stable-diffusion.cpp
                 into thirdparty/<project>/{include,lib}
    download     download a Llama or Whisper model for testing
    info         show pinned + checked-out dependency versions
    clean        remove build/ and thirdparty/<project>/{bin,lib,include}

Backend support (via build command flags or environment variables):
    --metal, -m       Enable Metal backend (macOS)
    --cuda, -c        Enable CUDA backend (NVIDIA GPUs)
    --vulkan, -V      Enable Vulkan backend (cross-platform)
    --sycl, -y        Enable SYCL backend (Intel GPUs)
    --hip, -H         Enable HIP/ROCm backend (AMD GPUs)
    --opencl, -o      Enable OpenCL backend
    --cpu-only, -C    Disable all GPU backends

Environment variables:
    GGML_METAL=1      Enable Metal backend (default ON on macOS)
    GGML_CUDA=1       Enable CUDA backend
    GGML_VULKAN=1     Enable Vulkan backend
    GGML_SYCL=1       Enable SYCL backend
    GGML_HIP=1        Enable HIP/ROCm backend
    GGML_OPENCL=1     Enable OpenCL backend
    SD_USE_VENDORED_GGML=0   Share llama.cpp's ggml with stable-diffusion.cpp
                             (required for chimera's static link).
"""

import argparse
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, NoReturn, Optional, TypeVar, Union
from urllib.request import urlretrieve

# ----------------------------------------------------------------------------
# type aliases

Pathlike = Union[str, Path]

# ----------------------------------------------------------------------------
# env helpers


def getenv(key: str, default: bool = False) -> bool:
    """Convert '0','1' env values to bool {True, False}."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return bool(int(value))
    except ValueError:
        logging.getLogger(__name__).warning(
            f"Invalid boolean value for {key}: {value}, using default {default}"
        )
        return default


def setenv(key: str, default: str) -> str:
    """get environ variable if it exists, else set default"""
    if key in os.environ:
        return os.getenv(key, default)
    os.environ[key] = default
    return default


# ----------------------------------------------------------------------------
# constants

PYTHON = sys.executable
PLATFORM = platform.system()
ARCH = platform.machine()
PY_VER_MINOR = sys.version_info.minor

# Version block. CMakeLists.txt parses these four constants out of this file
# to stamp the chimera binary at compile time. Keep names and "X = "Y"" form.
CHIMERA_VERSION = "0.1.4"
LLAMACPP_VERSION = "b9119"
WHISPERCPP_VERSION = "v1.8.4"
SDCPP_VERSION = "master-596-90e87bc"
# linenoise: shakfu's fork. No tags yet, so we pin a branch and record the
# commit in CHANGELOG for traceability.
LINENOISE_VERSION = "master"
# SQLite amalgamation; upstream files live at sqlite.org/<year>/
# sqlite-amalgamation-<digits>.zip, where the digits encode major*1000000 +
# minor*10000 + patch*100. 3470000 == 3.47.0.
SQLITE_VERSION = "3.47.0"
SQLITE_VERSION_URL_DIGITS = "3470000"
SQLITE_VERSION_URL_YEAR = "2024"
# sqlite-vec: vendored single-file extension built by chimera_db.cpp via
# sqlite3_vec_init. We pin a release tag for reproducibility.
SQLITE_VEC_VERSION = "v0.1.6"

if PLATFORM == "Darwin":
    MACOSX_DEPLOYMENT_TARGET = setenv("MACOSX_DEPLOYMENT_TARGET", "12.6")
DEBUG = getenv("DEBUG", default=True)
COLOR = getenv("COLOR", default=True)

# ----------------------------------------------------------------------------
# logging config


class CustomFormatter(logging.Formatter):
    """custom logging formatting class"""

    white = "\x1b[97;20m"
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - {}%(levelname)s{} - %(name)s.%(funcName)s - %(message)s"

    FORMATS = {
        logging.DEBUG: fmt.format(grey, reset),
        logging.INFO: fmt.format(green, reset),
        logging.WARNING: fmt.format(yellow, reset),
        logging.ERROR: fmt.format(red, reset),
        logging.CRITICAL: fmt.format(bold_red, reset),
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, handlers=[handler])

# ----------------------------------------------------------------------------
# utility classes


class ShellCmd:
    """Provides platform-agnostic file/folder + shell helpers."""

    log: logging.Logger

    def cmd(self, shellcmd: str, cwd: Pathlike = ".") -> None:
        """Run shell command within working directory.

        WARNING: shell=True for convenience. Only call with trusted input.
        """
        cwd_path = Path(cwd).resolve()
        self.log.info(shellcmd)
        try:
            subprocess.check_call(shellcmd, shell=True, cwd=str(cwd_path))
        except subprocess.CalledProcessError:
            self.log.critical("", exc_info=True)
            sys.exit(1)

    def download(
        self,
        url: str,
        tofolder: Optional[Pathlike] = None,
        max_size: int = 1024 * 1024 * 100,
    ) -> Pathlike:
        """Download a file from a url to an optional folder."""
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Unsupported URL scheme: {url}")
        basename = os.path.basename(url)
        if ".." in basename or basename.startswith("/"):
            raise ValueError(f"Invalid filename in URL: {url}")

        _path = Path(basename)
        if tofolder:
            _path = Path(tofolder).resolve().joinpath(_path)
            if _path.exists():
                return _path

        self.log.info(f"Downloading {url} to {_path}")
        filename, _ = urlretrieve(url, filename=_path)

        if _path.stat().st_size > max_size:
            _path.unlink()
            raise ValueError(
                f"Downloaded file exceeds size limit: {_path.stat().st_size} > {max_size}"
            )
        return Path(filename)

    def extract(self, archive: Pathlike, tofolder: Pathlike = ".") -> None:
        """Extract archive with path traversal protection."""
        tofolder_resolved = Path(tofolder).resolve()

        def safe_extract_tar(
            members: list[tarfile.TarInfo], dest: Path
        ) -> list[tarfile.TarInfo]:
            for member in members:
                member_path = (dest / member.name).resolve()
                if not str(member_path).startswith(str(dest)):
                    raise ValueError(f"Attempted path traversal in tar: {member.name}")
            return members

        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tar:
                safe_members = safe_extract_tar(tar.getmembers(), tofolder_resolved)
                tar.extractall(tofolder_resolved, members=safe_members)
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zip_file:
                for info in zip_file.infolist():
                    extracted_path = (tofolder_resolved / info.filename).resolve()
                    if not str(extracted_path).startswith(str(tofolder_resolved)):
                        raise ValueError(
                            f"Attempted path traversal in zip: {info.filename}"
                        )
                zip_file.extractall(tofolder_resolved)
        else:
            raise TypeError("cannot extract from this file.")

    def fail(self, msg: str, *args: object) -> NoReturn:
        """exits the program with an error msg."""
        self.log.critical(msg, *args)
        sys.exit(1)

    def git_clone(
        self,
        url: str,
        branch: Optional[str] = None,
        directory: Optional[Pathlike] = None,
        recurse: bool = False,
        cwd: Pathlike = ".",
    ) -> None:
        _cmds = ["git clone --depth 1"]
        if branch:
            _cmds.append(f"--branch {branch}")
        if recurse:
            _cmds.append("--recurse-submodules --shallow-submodules")
        _cmds.append(url)
        if directory:
            _cmds.append(str(directory))
        self.cmd(" ".join(_cmds), cwd=cwd)

    def copy(self, src: Pathlike, dst: Pathlike) -> None:
        """copy file or folder -- behaves like `cp -rf`."""
        self.log.info("copy %s to %s", src, dst)
        src, dst = Path(src), Path(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    def remove(self, path: Pathlike, silent: bool = False) -> None:
        """Remove file or folder."""

        def remove_readonly(func: Any, path: Any, exc_info: Any) -> None:
            if PY_VER_MINOR < 11:
                if func not in (os.unlink, os.rmdir) or exc_info[1].winerror != 5:
                    raise exc_info[1]
            else:
                if func not in (os.unlink, os.rmdir) or exc_info.winerror != 5:
                    raise exc_info
            os.chmod(path, stat.S_IWRITE)
            func(path)

        path = Path(path)
        if path.is_dir():
            if not silent:
                self.log.info("remove folder: %s", path)
            if PY_VER_MINOR < 11:
                shutil.rmtree(path, ignore_errors=not DEBUG, onerror=remove_readonly)
            else:
                shutil.rmtree(path, ignore_errors=not DEBUG, onexc=remove_readonly)  # type: ignore[call-arg]
        else:
            if not silent:
                self.log.info("remove file: %s", path)
            try:
                path.unlink()
            except FileNotFoundError:
                if not silent:
                    self.log.warning("file not found: %s", path)

    def glob_copy(
        self,
        src: Pathlike,
        dest: Pathlike,
        patterns: list[str],
    ) -> None:
        """copy glob patterns from src dir to destination dir"""
        src = Path(src)
        dest = Path(dest)
        if not src.exists():
            raise IOError(f"src dir '{src}' not found")
        if not dest.exists():
            dest.mkdir()
        for p in patterns:
            for f in src.glob(p):
                self.copy(f, dest)

    def cmake_config(
        self,
        src_dir: Pathlike,
        build_dir: Pathlike,
        *scripts: str,
        **options: Union[str, bool, int],
    ) -> None:
        """cmake configure / generate stage"""
        src_dir = Path(src_dir)
        build_dir = Path(build_dir)
        if not src_dir.exists():
            raise FileNotFoundError(f"CMake source directory not found: {src_dir}")
        build_dir.mkdir(parents=True, exist_ok=True)
        _cmds = [f"cmake -S {src_dir} -B {build_dir}"]
        if scripts:
            _cmds.append(" ".join(f"-C {path}" for path in scripts))
        if options:

            def cmake_value(v: Union[str, bool, int]) -> Union[str, int]:
                if isinstance(v, bool):
                    return "ON" if v else "OFF"
                return v

            def cmake_flag(k: str, v: Union[str, bool, int]) -> str:
                val = cmake_value(v)
                if isinstance(val, str) and ";" in val:
                    return f'-D{k}="{val}"'
                return f"-D{k}={val}"

            _cmds.append(" ".join(cmake_flag(k, v) for k, v in options.items()))
        self.cmd(" ".join(_cmds))

    def cmake_build(self, build_dir: Pathlike, release: bool = False) -> None:
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        _cmd += f" --parallel {os.cpu_count() or 4}"
        self.cmd(_cmd)

    def cmake_build_targets(
        self, build_dir: Pathlike, targets: list[str], release: bool = False
    ) -> None:
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        for target in targets:
            _cmd += f" --target {target}"
        _cmd += f" --parallel {os.cpu_count() or 4}"
        self.cmd(_cmd)

    def cmake_install(
        self, build_dir: Pathlike, prefix: Optional[Pathlike] = None
    ) -> None:
        _cmds = ["cmake --install", str(build_dir)]
        if prefix:
            _cmds.append(f"--prefix {str(prefix)}")
        self.cmd(" ".join(_cmds))


# ----------------------------------------------------------------------------
# project + builders


class Project(ShellCmd):
    """Holds chimera's directory layout."""

    cwd: Path
    build: Path
    src: Path
    thirdparty: Path
    install: Path
    scripts: Path

    def __init__(self) -> None:
        self.cwd = Path.cwd()
        self.build = self.cwd / "build"
        # Dep source trees live under build/ (separate from build/CMakeFiles).
        self.src = self.build
        self.thirdparty = self.cwd / "thirdparty"
        self.install = self.thirdparty
        self.scripts = self.cwd / "scripts"

    def setup(self) -> None:
        self.build.mkdir(exist_ok=True)
        self.src.mkdir(exist_ok=True)
        self.install.mkdir(exist_ok=True)


class AbstractBuilder(ShellCmd):
    """Common builder scaffolding."""

    name: str
    version: str
    repo_url: str
    libs: list[str]

    def __init__(
        self, version: Optional[str] = None, project: Optional[Project] = None
    ):
        self.version = version or self.version
        self.project = project or Project()
        self.log = logging.getLogger(self.__class__.__name__)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.name}-{self.version}'>"

    @property
    def src_dir(self) -> Path:
        return self.project.src / self.name

    @property
    def build_dir(self) -> Path:
        return self.src_dir / "build"

    @property
    def prefix(self) -> Path:
        return self.project.install / self.name.lower()

    @property
    def include(self) -> Path:
        return self.prefix / "include"

    @property
    def lib(self) -> Path:
        return self.prefix / "lib"

    def get_lib_path(self, build_dir: Path, subdir: str, name: str) -> Path:
        """Platform-specific library path from a CMake build directory."""
        base = build_dir / subdir
        if PLATFORM == "Windows":
            release_path = base / "Release" / f"{name}.lib"
            if release_path.exists():
                return release_path
            direct_path = base / f"{name}.lib"
            if direct_path.exists():
                return direct_path
            for config in ("RelWithDebInfo", "MinSizeRel", "Debug"):
                config_path = base / config / f"{name}.lib"
                if config_path.exists():
                    self.log.warning(
                        f"Library {name}.lib not found in Release/ or root, using {config}/ build"
                    )
                    return config_path
            self.log.warning(
                f"Library {name}.lib not found in any configuration under {base}. "
                f"Searched: Release/, direct, RelWithDebInfo/, MinSizeRel/, Debug/"
            )
            return release_path
        return base / f"lib{name}.a"

    def copy_lib(
        self,
        build_dir: Path,
        subdir: str,
        name: str,
        dest: Path,
        required: bool = True,
    ) -> bool:
        lib_path = self.get_lib_path(build_dir, subdir, name)
        if lib_path.exists():
            self.copy(lib_path, dest)
            self.log.info(f"Copied {lib_path.name} to {dest}")
            return True
        if required:
            raise FileNotFoundError(f"Required library not found: {lib_path}")
        self.log.warning(f"Optional library not found: {lib_path}")
        return False


class Builder(AbstractBuilder):
    """Concrete builder: clones source from repo_url at self.version."""

    def setup(self) -> None:
        self.log.info(f"update from {self.name} main repo")
        self.project.setup()
        if self.version:
            self.git_clone(
                self.repo_url, branch=self.version, recurse=True, cwd=self.project.src
            )
        else:
            self.git_clone(self.repo_url, recurse=True, cwd=self.project.src)


class GgmlBuilder(Builder):
    """Builder base for ggml-backed projects (llama.cpp / whisper.cpp / sd.cpp).

    Shared helpers for mapping GGML_* env flags onto CMake options.
    Subclasses must implement `get_backend_cmake_options`.
    """

    base_libs: list[str] = ["ggml"]
    extra_libs: list[str] = []

    @property
    def libs(self) -> list[str]:  # type: ignore[override]
        return list(self.base_libs) + list(self.extra_libs)

    def get_backend_cmake_options(self) -> dict[str, Any]:
        raise NotImplementedError

    CUDA_TUNING_ENV_FLAGS: tuple[str, ...] = (
        "GGML_CUDA_FORCE_MMQ",
        "GGML_CUDA_FORCE_CUBLAS",
        "GGML_CUDA_PEER_MAX_BATCH_SIZE",
        "GGML_CUDA_FA_ALL_QUANTS",
    )

    BACKEND_SHORT_NAMES: dict[str, str] = {
        "GGML_METAL": "metal",
        "GGML_CUDA": "cuda",
        "GGML_VULKAN": "vulkan",
        "GGML_SYCL": "sycl",
        "GGML_HIP": "hip",
        "GGML_OPENCL": "opencl",
    }

    def enabled_backends_from_env(self) -> list[str]:
        result = []
        for env_name, short in self.BACKEND_SHORT_NAMES.items():
            default = env_name == "GGML_METAL" and PLATFORM == "Darwin"
            if getenv(env_name, default=default):
                result.append(short)
        return result

    def _forward_env_flags(self, options: dict[str, Any], names: Iterable[str]) -> None:
        for name in names:
            val = os.environ.get(name)
            if val is not None:
                options[name] = val
                self.log.info(f"  {name}={val}")

    def _set_backend(
        self,
        options: dict[str, Any],
        cmake_name: str,
        enabled: bool,
        label: str,
        suffix: str = "",
    ) -> None:
        options[cmake_name] = "ON" if enabled else "OFF"
        if enabled:
            self.log.info(f"Enabling {label} backend{suffix}")

    def _apply_cuda_extras(self, options: dict[str, Any]) -> None:
        cuda_archs = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
        if cuda_archs:
            options["CMAKE_CUDA_ARCHITECTURES"] = cuda_archs
            self.log.info(f"  CUDA architectures: {cuda_archs}")
        cuda_compiler = os.environ.get("CMAKE_CUDA_COMPILER")
        if cuda_compiler:
            options["CMAKE_CUDA_COMPILER"] = cuda_compiler
            self.log.info(f"  CUDA compiler: {cuda_compiler}")
        self._forward_env_flags(options, self.CUDA_TUNING_ENV_FLAGS)

    def _apply_hip_archs(self, options: dict[str, Any]) -> None:
        hip_archs = os.environ.get("CMAKE_HIP_ARCHITECTURES")
        if hip_archs:
            options["CMAKE_HIP_ARCHITECTURES"] = hip_archs
            self.log.info(f"  HIP architectures: {hip_archs}")

    def _apply_openmp(self, options: dict[str, Any]) -> None:
        openmp = os.environ.get("GGML_OPENMP")
        if openmp is not None:
            options["GGML_OPENMP"] = "ON" if openmp == "1" else "OFF"
            self.log.info(f"  GGML_OPENMP={options['GGML_OPENMP']}")


class LlamaCppBuilder(GgmlBuilder):
    """Build llama.cpp into thirdparty/llama.cpp/{include,lib}."""

    name: str = "llama.cpp"
    version: str = LLAMACPP_VERSION
    repo_url: str = "https://github.com/ggml-org/llama.cpp.git"
    base_libs: list[str] = ["ggml", "ggml-base", "ggml-cpu"]
    extra_libs: list[str] = [
        "llama",
        "llama-common",
        "mtmd",
        "server-context",
        "cpp-httplib",
    ]

    def get_backend_cmake_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}

        metal = getenv("GGML_METAL", default=(PLATFORM == "Darwin"))
        cuda = getenv("GGML_CUDA", default=False)
        vulkan = getenv("GGML_VULKAN", default=False)
        sycl = getenv("GGML_SYCL", default=False)
        hip = getenv("GGML_HIP", default=False)
        opencl = getenv("GGML_OPENCL", default=False)

        self._set_backend(options, "GGML_METAL", metal, "Metal")
        self._set_backend(options, "GGML_CUDA", cuda, "CUDA")
        if cuda:
            self._apply_cuda_extras(options)
        self._set_backend(options, "GGML_VULKAN", vulkan, "Vulkan")
        self._set_backend(options, "GGML_SYCL", sycl, "SYCL")
        self._set_backend(options, "GGML_HIP", hip, "HIP/ROCm")
        if hip:
            self._apply_hip_archs(options)
            if getenv("GGML_HIP_ROCWMMA_FATTN", default=False):
                options["GGML_HIP_ROCWMMA_FATTN"] = "ON"
                self.log.info("  rocWMMA flash attention enabled")
        self._set_backend(options, "GGML_OPENCL", opencl, "OpenCL")

        if getenv("GGML_BLAS", default=False):
            options["GGML_BLAS"] = "ON"
            blas_vendor = os.environ.get("GGML_BLAS_VENDOR")
            if blas_vendor:
                options["GGML_BLAS_VENDOR"] = blas_vendor
                self.log.info(f"Enabling BLAS backend (vendor: {blas_vendor})")
            else:
                self.log.info("Enabling BLAS backend")

        self._apply_openmp(options)

        ggml_native = os.environ.get("GGML_NATIVE")
        if ggml_native is not None:
            options["GGML_NATIVE"] = "ON" if ggml_native == "1" else "OFF"
            self.log.info(f"  GGML_NATIVE={options['GGML_NATIVE']}")

        return options

    def copy_backend_libs(self) -> None:
        """Copy backend-specific static libs (metal/cuda/vulkan/...) into self.lib."""
        enabled = self.enabled_backends_from_env()
        # Metal builds also need the BLAS backend lib alongside ggml-metal.
        if "metal" in enabled:
            self.copy_lib(self.build_dir, "ggml/src/ggml-blas", "ggml-blas", self.lib)
        for short in enabled:
            self.copy_lib(
                self.build_dir, f"ggml/src/ggml-{short}", f"ggml-{short}", self.lib
            )

    def _copy_headers(self) -> None:
        """Copy llama.cpp public headers into the prefix include dir."""
        self.glob_copy(self.src_dir / "common", self.include, patterns=["*.h", "*.hpp"])
        self.glob_copy(
            self.src_dir / "ggml" / "include", self.include, patterns=["*.h"]
        )
        self.glob_copy(self.src_dir / "include", self.include, patterns=["*.h"])
        # jinja headers (required by chat.h).
        jinja_include = self.include / "jinja"
        jinja_include.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "common" / "jinja", jinja_include, patterns=["*.h", "*.hpp"]
        )
        # nlohmann JSON headers (required by json-partial.h).
        nlohmann_include = self.include / "nlohmann"
        nlohmann_include.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "vendor" / "nlohmann", nlohmann_include, patterns=["*.hpp"]
        )
        # mtmd (multimodal) headers.
        self.glob_copy(self.src_dir / "tools" / "mtmd", self.include, patterns=["*.h"])
        # server-context (OpenAI-compatible engine) headers. server-context
        # itself is a STATIC library upstream; server-http is *not* — it lives
        # only in llama-server's own source list, so its .cpp is also copied
        # below as an auxiliary source the consumer (chimera) compiles into
        # its own target.
        self.glob_copy(
            self.src_dir / "tools" / "server", self.include, patterns=["server-*.h"]
        )
        # cpp-httplib (single-header). server-http.cpp includes it as
        # `<cpp-httplib/httplib.h>`, so the header goes under a subdir of
        # the same name, not at the include root.
        httplib_src = self.src_dir / "vendor" / "cpp-httplib"
        httplib_dst = self.include / "cpp-httplib"
        if httplib_src.exists():
            httplib_dst.mkdir(exist_ok=True)
            self.glob_copy(httplib_src, httplib_dst, patterns=["httplib.h"])
        # Auxiliary source files: server-http.cpp is not part of
        # libserver-context.a (upstream compiles it directly into the
        # llama-server binary). Stash it under src-aux/ so chimera's
        # CMakeLists can compile it as part of the chimera target.
        src_aux = self.prefix / "src-aux"
        src_aux.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "tools" / "server", src_aux, patterns=["server-http.cpp"]
        )
        # Webui assets + xxd helper. server-http.cpp #includes
        # "index.html.hpp" / "bundle.{js,css}.hpp" / "loading.html.hpp"
        # when compiled with LLAMA_BUILD_WEBUI defined. These .hpp files
        # are xxd-baked from the raw assets at chimera build time when
        # the top-level CHIMERA_WEBUI_EMBED option is ON; staging the
        # raw assets and the xxd helper here means the option flip
        # doesn't require re-running the llama.cpp builder. The asset
        # bundle is ~7 MB total — tolerable footprint for a dev tree
        # even when the feature isn't enabled. Pre-built (no Node
        # toolchain needed) so the chimera build can stay self-contained.
        webui_aux = src_aux / "webui"
        webui_aux.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "tools" / "server" / "public",
            webui_aux,
            patterns=["index.html", "bundle.js", "bundle.css", "loading.html"],
        )
        self.copy(self.src_dir / "scripts" / "xxd.cmake", src_aux / "xxd.cmake")

    def build(self) -> None:
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self._copy_headers()

        backend_options = self.get_backend_cmake_options()

        # SD requires GGML_MAX_NAME=128 (vs llama.cpp's 64). When both share
        # the same ggml, both sides must agree or ggml_tensor's layout diverges.
        extra = {}
        if StableDiffusionCppBuilder.uses_shared_ggml():
            _def = f"-DGGML_MAX_NAME={StableDiffusionCppBuilder.GGML_MAX_NAME}"
            extra["CMAKE_C_FLAGS"] = _def
            extra["CMAKE_CXX_FLAGS"] = _def

        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            LLAMA_CURL=False,
            LLAMA_OPENSSL=True,
            # LLAMA_BUILD_SERVER=ON adds tools/server to the cmake graph so the
            # server-context STATIC library target is defined. We do NOT build
            # the llama-server *executable* below; only the static lib and the
            # vendored cpp-httplib are needed to power `chimera serve`. The
            # webui route binding lives in server-http.cpp, which chimera
            # compiles itself (it's not part of libserver-context.a). So
            # LLAMA_BUILD_WEBUI here stays OFF — the asset .hpp files and
            # the LLAMA_BUILD_WEBUI compile define are produced on the chimera
            # side instead, gated by the top-level CHIMERA_WEBUI_EMBED CMake
            # option. _copy_headers() unconditionally stages the prebuilt
            # public/* assets and xxd.cmake into src-aux/ so the option can
            # flip them on without re-running this builder.
            LLAMA_BUILD_SERVER=True,
            LLAMA_BUILD_WEBUI=False,
            LLAMA_BUILD_TESTS=False,
            LLAMA_BUILD_EXAMPLES=False,
            **extra,
            **backend_options,
        )
        # Build only the targets chimera needs. Notably, the llama-server
        # executable is intentionally absent — we link server-context as a
        # library and provide our own (smaller) entry point in chimera_serve.
        self.cmake_build_targets(
            build_dir=self.build_dir,
            targets=["llama", "llama-common", "mtmd", "server-context", "cpp-httplib"],
            release=True,
        )

        self.lib.mkdir(parents=True, exist_ok=True)
        self.copy_lib(self.build_dir, "common", "llama-common", self.lib)
        # cpp-httplib is now a hard dep (was optional when SERVER=OFF). Drop
        # required=False so a missing artifact becomes a visible build error
        # rather than a silent skip that explodes at chimera link time.
        self.copy_lib(self.build_dir, "vendor/cpp-httplib", "cpp-httplib", self.lib)
        self.copy_lib(self.build_dir, "src", "llama", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml-base", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml-cpu", self.lib)
        self.copy_lib(self.build_dir, "tools/mtmd", "mtmd", self.lib)
        self.copy_lib(self.build_dir, "tools/server", "server-context", self.lib)
        self.copy_backend_libs()


class WhisperCppBuilder(GgmlBuilder):
    """Build whisper.cpp into thirdparty/whisper.cpp/{include,lib}."""

    name: str = "whisper.cpp"
    version: str = WHISPERCPP_VERSION
    repo_url: str = "https://github.com/ggml-org/whisper.cpp"
    base_libs: list[str] = ["ggml"]
    extra_libs: list[str] = ["whisper", "common"]

    def get_backend_cmake_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        sfx = " for whisper.cpp"

        metal = (
            getenv("GGML_METAL", default=(PLATFORM == "Darwin"))
            and PLATFORM == "Darwin"
        )
        cuda = getenv("GGML_CUDA", default=False)
        vulkan = getenv("GGML_VULKAN", default=False)
        sycl = getenv("GGML_SYCL", default=False)
        hip = getenv("GGML_HIP", default=False)
        opencl = getenv("GGML_OPENCL", default=False)

        self._set_backend(options, "GGML_METAL", metal, "Metal", sfx)
        self._set_backend(options, "GGML_CUDA", cuda, "CUDA", sfx)
        if cuda:
            self._apply_cuda_extras(options)
        self._set_backend(options, "GGML_VULKAN", vulkan, "Vulkan", sfx)
        self._set_backend(options, "GGML_SYCL", sycl, "SYCL", sfx)
        self._set_backend(options, "GGML_HIP", hip, "HIP/ROCm", sfx)
        if hip:
            self._apply_hip_archs(options)
            if getenv("GGML_HIP_ROCWMMA_FATTN", default=False):
                options["GGML_HIP_ROCWMMA_FATTN"] = "ON"
        self._set_backend(options, "GGML_OPENCL", opencl, "OpenCL", sfx)

        if getenv("GGML_BLAS", default=False):
            options["GGML_BLAS"] = "ON"
            blas_vendor = os.environ.get("GGML_BLAS_VENDOR")
            if blas_vendor:
                options["GGML_BLAS_VENDOR"] = blas_vendor
            self.log.info(f"Enabling BLAS backend{sfx}")

        self._apply_openmp(options)
        return options

    def build(self) -> None:
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self.glob_copy(
            self.src_dir / "examples", self.include, patterns=["*.h", "*.hpp"]
        )

        backend_options = self.get_backend_cmake_options()

        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            CMAKE_INSTALL_LIBDIR="lib",  # avoid lib64 on 64-bit Linux
            **backend_options,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy_lib(self.build_dir, "examples", "common", self.lib)


class StableDiffusionCppBuilder(GgmlBuilder):
    """Build stable-diffusion.cpp into thirdparty/stable-diffusion.cpp/{include,lib}."""

    name: str = "stable-diffusion.cpp"
    version: str = SDCPP_VERSION
    repo_url: str = "https://github.com/leejet/stable-diffusion.cpp.git"
    base_libs: list[str] = ["stable-diffusion"]
    extra_libs: list[str] = []

    # SD requires GGML_MAX_NAME=128 (its CMakeLists.txt:233, ggml_extend.hpp:94).
    # llama.cpp defaults to 64. When SD shares llama.cpp's ggml
    # (SD_USE_VENDORED_GGML=0, chimera's default), both sides must agree.
    GGML_MAX_NAME: int = 128

    @staticmethod
    def uses_shared_ggml() -> bool:
        return os.environ.get("SD_USE_VENDORED_GGML") == "0"

    def get_backend_cmake_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        sfx = " for stable-diffusion.cpp"

        metal = (
            getenv("GGML_METAL", default=(PLATFORM == "Darwin"))
            and PLATFORM == "Darwin"
        )
        cuda = getenv("GGML_CUDA", default=False)
        vulkan = getenv("GGML_VULKAN", default=False)
        sycl = getenv("GGML_SYCL", default=False)
        hip = getenv("GGML_HIP", default=False)
        opencl = getenv("GGML_OPENCL", default=False)

        self._set_backend(options, "SD_METAL", metal, "Metal", sfx)
        self._set_backend(options, "SD_CUDA", cuda, "CUDA", sfx)
        if cuda:
            self._apply_cuda_extras(options)
        self._set_backend(options, "SD_VULKAN", vulkan, "Vulkan", sfx)
        self._set_backend(options, "SD_SYCL", sycl, "SYCL", sfx)
        self._set_backend(options, "SD_HIPBLAS", hip, "HIP/ROCm", sfx)
        if hip:
            self._apply_hip_archs(options)
        self._set_backend(options, "SD_OPENCL", opencl, "OpenCL", sfx)

        self._apply_openmp(options)
        return options

    def _sync_ggml_abi(self) -> None:
        """Replace SD's vendored ggml with llama.cpp's so enum values agree.

        SD vendors its own ggml. When chimera links SD against llama.cpp's
        ggml, ggml_op / ggml_type ids must match between header and runtime
        or compute graphs build with wrong op ids and assert at runtime.
        """
        llama_ggml = self.project.src / "llama.cpp" / "ggml"
        sd_ggml = self.src_dir / "ggml"
        if not llama_ggml.exists() or not sd_ggml.exists():
            self.log.warning("Cannot sync ggml ABI: llama.cpp or SD ggml dir missing")
            return
        shutil.rmtree(sd_ggml)
        shutil.copytree(llama_ggml, sd_ggml)
        self.log.info(
            "Replaced SD's vendored ggml with llama.cpp's ggml for ABI compatibility"
        )

    def build(self, examples: bool = True) -> None:
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")

        # Sync ggml ABI from llama.cpp before compiling so SD and llama.cpp
        # use the same ggml_op / ggml_type values. Only needed when SD links
        # against llama.cpp's ggml (chimera's default, SD_USE_VENDORED_GGML=0).
        if os.environ.get("SD_USE_VENDORED_GGML") == "0":
            self._sync_ggml_abi()

        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self.glob_copy(self.src_dir, self.include, patterns=["*.h", "*.hpp"])
        # stb headers for zero-dependency image I/O
        stb_src = self.src_dir / "thirdparty"
        if stb_src.exists():
            for stb_file in ["stb_image.h", "stb_image_write.h", "stb_image_resize.h"]:
                stb_path = stb_src / stb_file
                if stb_path.exists():
                    self.copy(stb_path, self.include)
                    self.log.info(f"Copied {stb_file} to include directory")

        backend_options = self.get_backend_cmake_options()

        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            CMAKE_INSTALL_LIBDIR="lib",
            SD_BUILD_EXAMPLES=examples,
            **backend_options,
        )
        self.cmake_build(build_dir=self.build_dir, release=True)
        self.cmake_install(build_dir=self.build_dir, prefix=self.prefix)
        self.copy_lib(self.build_dir, ".", "stable-diffusion", self.lib)


class LinenoiseBuilder(Builder):
    """Build shakfu's linenoise fork into thirdparty/linenoise/{include,lib}.

    Not ggml-backed: a plain C99 line-editor library. We disable the
    examples and tree-sitter syntax highlighting so the build has no
    extra deps.
    """

    name: str = "linenoise"
    version: str = LINENOISE_VERSION
    repo_url: str = "https://github.com/shakfu/linenoise.git"
    libs: list[str] = ["linenoise"]

    def build(self) -> None:
        if not self.src_dir.exists():
            self.setup()
        self.log.info(f"building {self.name}")
        self.prefix.mkdir(exist_ok=True)
        self.include.mkdir(exist_ok=True)
        self.glob_copy(self.src_dir / "include", self.include, patterns=["*.h"])

        # BUILD_TESTS=OFF: upstream's test block references linenoise-example
        # in `set_tests_properties(... DEPENDS linenoise-example)`, so
        # configure fails when examples are disabled but tests remain on.
        # We disable both so neither the example nor the test suite is
        # configured. We then only build the `linenoise` target explicitly.
        self.cmake_config(
            src_dir=self.src_dir,
            build_dir=self.build_dir,
            BUILD_SHARED_LIBS=False,
            BUILD_EXAMPLES=False,
            BUILD_TESTS=False,
            WITH_TREESITTER=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
        )
        self.cmake_build_targets(
            build_dir=self.build_dir, targets=["linenoise"], release=True
        )

        self.lib.mkdir(parents=True, exist_ok=True)
        self.copy_lib(self.build_dir, ".", "linenoise", self.lib)


class SqliteBuilder(AbstractBuilder):
    """Vendor SQLite as a single-translation-unit amalgamation.

    Unlike the other builders this one neither configures nor compiles
    anything: SQLite is shipped as plain C (sqlite3.c + sqlite3.h) and
    chimera's own CMake target compiles it inline (the same pattern we
    use for tools/server/server-http.cpp). manage.py here is only
    responsible for fetching the amalgamation and dropping the two
    files into thirdparty/sqlite/{include,src-aux}/.
    """

    name: str = "sqlite"
    version: str = SQLITE_VERSION
    repo_url: str = ""  # not git; we fetch the .zip from sqlite.org
    libs: list[str] = []

    def setup(self) -> None:
        # We download lazily inside build(); nothing to do upfront beyond
        # ensuring the project root exists.
        self.project.setup()

    def build(self) -> None:
        self.log.info(f"fetching {self.name} {self.version}")
        self.prefix.mkdir(parents=True, exist_ok=True)
        self.include.mkdir(exist_ok=True)
        src_aux = self.prefix / "src-aux"
        src_aux.mkdir(exist_ok=True)

        # If the destination files already exist with the pinned version,
        # skip the download — manage.py re-runs on every `make deps`.
        sentinel = self.prefix / f".version-{self.version}"
        if sentinel.exists():
            self.log.info(f"{self.name} {self.version} already vendored")
            return

        zip_name = f"sqlite-amalgamation-{SQLITE_VERSION_URL_DIGITS}.zip"
        url = f"https://sqlite.org/{SQLITE_VERSION_URL_YEAR}/{zip_name}"
        stage = self.project.src / self.name
        stage.mkdir(parents=True, exist_ok=True)
        archive = self.download(url, tofolder=stage)
        self.extract(archive, tofolder=stage)

        extracted = stage / f"sqlite-amalgamation-{SQLITE_VERSION_URL_DIGITS}"
        if not extracted.exists():
            raise FileNotFoundError(
                f"expected extracted directory not found: {extracted}"
            )

        # The amalgamation ships sqlite3.c + sqlite3.h (plus shell.c +
        # sqlite3ext.h which we don't need for embedded use).
        self.copy(extracted / "sqlite3.h", self.include / "sqlite3.h")
        # sqlite3ext.h is also needed when loading extensions like
        # sqlite-vec via the sqlite3_load_extension API path. We don't go
        # via load_extension (we statically link sqlite3_vec_init), but
        # sqlite-vec.c #includes "sqlite3ext.h" so the header must be
        # present alongside the source.
        self.copy(extracted / "sqlite3ext.h", self.include / "sqlite3ext.h")
        self.copy(extracted / "sqlite3.c", src_aux / "sqlite3.c")

        sentinel.write_text(self.version)
        self.log.info(f"vendored {self.name} {self.version}")


class SqliteVecBuilder(AbstractBuilder):
    """Vendor sqlite-vec (asg017/sqlite-vec) as a single-TU amalgamation.

    Same shape as SqliteBuilder: no library is built; chimera compiles
    the source file into its own target.
    """

    name: str = "sqlite-vec"
    version: str = SQLITE_VEC_VERSION
    repo_url: str = "https://github.com/asg017/sqlite-vec.git"
    libs: list[str] = []

    def setup(self) -> None:
        self.project.setup()
        if self.src_dir.exists():
            return
        # Shallow clone at the pinned tag.
        self.git_clone(self.repo_url, branch=self.version, cwd=self.project.src)

    def build(self) -> None:
        self.log.info(f"fetching {self.name} {self.version}")
        if not self.src_dir.exists():
            self.setup()
        self.prefix.mkdir(parents=True, exist_ok=True)
        self.include.mkdir(exist_ok=True)
        src_aux = self.prefix / "src-aux"
        src_aux.mkdir(exist_ok=True)

        sentinel = self.prefix / f".version-{self.version}"
        if sentinel.exists():
            self.log.info(f"{self.name} {self.version} already vendored")
            return

        c_src = self.src_dir / "sqlite-vec.c"
        if not c_src.exists():
            raise FileNotFoundError(f"sqlite-vec.c not found at {c_src}")

        # sqlite-vec ships sqlite-vec.h.tmpl rather than a prebuilt header;
        # upstream's CMake fills the ${VERSION_*} placeholders at build
        # time. We expand them here so chimera's compile sees a concrete
        # sqlite-vec.h. The template variables we care about all derive
        # from the pinned tag.
        tmpl_path = self.src_dir / "sqlite-vec.h.tmpl"
        if not tmpl_path.exists():
            raise FileNotFoundError(f"sqlite-vec.h.tmpl not found at {tmpl_path}")
        tag = self.version.lstrip("v")  # "v0.1.6" -> "0.1.6"
        parts = tag.split(".")
        while len(parts) < 3:
            parts.append("0")
        substitutions = {
            "${VERSION}": tag,
            "${VERSION_MAJOR}": parts[0],
            "${VERSION_MINOR}": parts[1],
            "${VERSION_PATCH}": parts[2],
            "${DATE}": "chimera-vendored",
            "${SOURCE}": f"asg017/sqlite-vec@{self.version}",
        }
        header = tmpl_path.read_text()
        for k, v in substitutions.items():
            header = header.replace(k, v)
        (self.include / "sqlite-vec.h").write_text(header)
        self.copy(c_src, src_aux / "sqlite-vec.c")

        sentinel.write_text(self.version)
        self.log.info(f"vendored {self.name} {self.version}")


# ----------------------------------------------------------------------------
# argparse decorator scaffolding

_F = TypeVar("_F", bound=Callable[..., Any])


def option(*args: Any, **kwds: Any) -> Callable[[_F], _F]:
    def _decorator(func: _F) -> _F:
        _option = (args, kwds)
        if hasattr(func, "options"):
            func.options.append(_option)
        else:
            func.options = [_option]  # type: ignore[attr-defined]
        return func

    return _decorator


def opt(long: str, short: str, desc: str, **kwargs: Any) -> Callable[[_F], _F]:
    return option(long, short, help=desc, action="store_true", **kwargs)


class MetaCommander(type):
    def __new__(
        cls,
        classname: str,
        bases: tuple[type, ...],
        classdict: dict[str, Any],
    ) -> "MetaCommander":
        classdict = dict(classdict)
        subcmds: dict[str, dict[str, Any]] = {}
        for name, func in list(classdict.items()):
            if name.startswith("do_"):
                name = name[3:]
                subcmd: dict[str, Any] = {"name": name, "func": func, "options": []}
                if hasattr(func, "options"):
                    subcmd["options"] = func.options
                subcmds[name] = subcmd
        classdict["_argparse_subcmds"] = subcmds
        return type.__new__(cls, classname, bases, classdict)


class Application(ShellCmd, metaclass=MetaCommander):
    """chimera build manager"""

    version: str = CHIMERA_VERSION
    epilog: str = ""
    default_args: list[str] = ["--help"]
    project: Project
    parser: argparse.ArgumentParser
    options: argparse.Namespace
    _argparse_subcmds: dict[str, Any]

    def __init__(self) -> None:
        self.project = Project()
        self.log = logging.getLogger(self.__class__.__name__)

    def parse_args(self) -> argparse.ArgumentParser:
        return argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=self.__doc__,
            epilog=self.epilog,
        )

    def cmdline(self) -> None:
        self.parser = self.parse_args()
        self.parser.add_argument(
            "-v", "--version", action="version", version="%(prog)s " + self.version
        )
        subparsers = self.parser.add_subparsers(
            title="subcommands",
            description="valid subcommands",
            help="additional help",
            metavar="",
        )

        for name in sorted(self._argparse_subcmds.keys()):
            subcmd = self._argparse_subcmds[name]
            subparser = subparsers.add_parser(
                subcmd["name"], help=subcmd["func"].__doc__
            )
            for args, kwds in subcmd["options"]:
                subparser.add_argument(*args, **kwds)
            subparser.set_defaults(func=subcmd["func"])

        if len(sys.argv) <= 1:
            options = self.parser.parse_args(self.default_args)
        else:
            options = self.parser.parse_args()

        self.options = options
        options.func(self, options)

    # ------------------------------------------------------------------------
    # build

    @opt("--metal", "-m", "enable Metal backend (macOS)")
    @opt("--cuda", "-c", "enable CUDA backend (NVIDIA GPUs)")
    @opt("--vulkan", "-V", "enable Vulkan backend (cross-platform)")
    @opt("--sycl", "-y", "enable SYCL backend (Intel GPUs)")
    @opt("--hip", "-H", "enable HIP/ROCm backend (AMD GPUs)")
    @opt("--opencl", "-o", "enable OpenCL backend")
    @option(
        "--blas",
        help="enable BLAS backend (use GGML_BLAS_VENDOR env var for vendor)",
        action="store_true",
    )
    @option("--no-openmp", help="disable OpenMP", action="store_true")
    @opt("--cpu-only", "-C", "disable all GPU backends (CPU only)")
    @opt("-l", "--llama-cpp", "build llama.cpp only")
    @opt("-w", "--whisper-cpp", "build whisper.cpp only")
    @opt("-d", "--stable-diffusion", "build stable-diffusion.cpp only")
    @opt("-L", "--linenoise", "build linenoise only")
    @opt("-q", "--sqlite", "vendor SQLite amalgamation only")
    @opt("-Q", "--sqlite-vec", "vendor sqlite-vec amalgamation only")
    @opt("-a", "--all", "build all dependencies")
    @opt("-D", "--deps-only", "(retained for Makefile compatibility; no-op)")
    @option(
        "--no-sd-examples",
        help="skip building stable-diffusion.cpp examples (sd-cli, sd-server)",
        action="store_true",
    )
    @option(
        "--sd-shared-ggml",
        help="link stable-diffusion against llama.cpp's shared ggml "
        "(required for chimera; sets SD_USE_VENDORED_GGML=0)",
        action="store_true",
    )
    @option(
        "--llama-version",
        default=LLAMACPP_VERSION,
        help=f"llama.cpp version (default: {LLAMACPP_VERSION})",
    )
    @option(
        "--whisper-version",
        default=WHISPERCPP_VERSION,
        help=f"whisper.cpp version (default: {WHISPERCPP_VERSION})",
    )
    @option(
        "--sd-version",
        default=SDCPP_VERSION,
        help=f"stable-diffusion.cpp version (default: {SDCPP_VERSION})",
    )
    @option(
        "--linenoise-version",
        default=LINENOISE_VERSION,
        help=f"linenoise version/branch (default: {LINENOISE_VERSION})",
    )
    @option(
        "--sqlite-version",
        default=SQLITE_VERSION,
        help=f"SQLite version (default: {SQLITE_VERSION})",
    )
    @option(
        "--sqlite-vec-version",
        default=SQLITE_VEC_VERSION,
        help=f"sqlite-vec tag (default: {SQLITE_VEC_VERSION})",
    )
    def do_build(self, args: argparse.Namespace) -> None:
        """fetch + build third-party dependencies into thirdparty/<project>/"""
        if args.cpu_only:
            for k in (
                "GGML_METAL",
                "GGML_CUDA",
                "GGML_VULKAN",
                "GGML_SYCL",
                "GGML_HIP",
                "GGML_OPENCL",
            ):
                os.environ[k] = "0"
        else:
            if args.metal:
                os.environ["GGML_METAL"] = "1"
            if args.cuda:
                os.environ["GGML_CUDA"] = "1"
            if args.vulkan:
                os.environ["GGML_VULKAN"] = "1"
            if args.sycl:
                os.environ["GGML_SYCL"] = "1"
            if args.hip:
                os.environ["GGML_HIP"] = "1"
            if args.opencl:
                os.environ["GGML_OPENCL"] = "1"
            if args.blas:
                os.environ["GGML_BLAS"] = "1"
        if args.no_openmp:
            os.environ["GGML_OPENMP"] = "0"

        if args.sd_shared_ggml:
            os.environ["SD_USE_VENDORED_GGML"] = "0"

        builder_versions = {
            LlamaCppBuilder: args.llama_version,
            WhisperCppBuilder: args.whisper_version,
            StableDiffusionCppBuilder: args.sd_version,
            LinenoiseBuilder: args.linenoise_version,
            SqliteBuilder: args.sqlite_version,
            SqliteVecBuilder: args.sqlite_vec_version,
        }

        _builders: list[type[AbstractBuilder]] = []
        if args.all:
            _builders = [
                LlamaCppBuilder,
                WhisperCppBuilder,
                StableDiffusionCppBuilder,
                LinenoiseBuilder,
                SqliteBuilder,
                SqliteVecBuilder,
            ]
        else:
            if args.llama_cpp:
                _builders.append(LlamaCppBuilder)
            if args.whisper_cpp:
                _builders.append(WhisperCppBuilder)
            if args.stable_diffusion:
                _builders.append(StableDiffusionCppBuilder)
            if args.linenoise:
                _builders.append(LinenoiseBuilder)
            if args.sqlite:
                _builders.append(SqliteBuilder)
            if args.sqlite_vec:
                _builders.append(SqliteVecBuilder)

        if not _builders:
            self.log.error(
                "No builders selected; pass --all or one of --llama-cpp / "
                "--whisper-cpp / --stable-diffusion / --sqlite / --sqlite-vec"
            )
            return

        for BuilderClass in _builders:
            version = builder_versions.get(BuilderClass)
            builder = BuilderClass(version=version)
            kwargs: dict[str, Any] = {}
            if isinstance(builder, StableDiffusionCppBuilder) and args.no_sd_examples:
                kwargs["examples"] = False
            builder.build(**kwargs)

    # ------------------------------------------------------------------------
    # info

    def do_info(self, args: argparse.Namespace) -> None:
        """show pinned + checked-out dependency versions"""
        pinned = {
            "llama.cpp": LLAMACPP_VERSION,
            "whisper.cpp": WHISPERCPP_VERSION,
            "stable-diffusion.cpp": SDCPP_VERSION,
            "linenoise": LINENOISE_VERSION,
        }
        print(f"chimera:               {CHIMERA_VERSION}")
        for name, ver in pinned.items():
            print(f"{name + ':':22s} pinned={ver}", end="")
            src_dir = self.project.src / name
            if not src_dir.exists():
                print("  (not cloned)")
                continue
            try:
                short = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=src_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                tag = (
                    subprocess.run(
                        ["git", "tag", "--points-at", "HEAD"],
                        cwd=src_dir,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    .stdout.strip()
                    .split("\n")[0]
                )
                if tag:
                    print(f"  checked-out=tag:{tag} ({short})")
                else:
                    print(f"  checked-out=commit:{short}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  (unable to read git state)")

    # ------------------------------------------------------------------------
    # bump-check
    #
    # Headers in tools/server/ (server-context.h, server-http.h) are NOT part
    # of upstream llama.cpp's stable public API; they're internal to the
    # llama-server build and shift with refactors. chimera links the
    # libserver-context.a static library + compiles server-http.cpp directly,
    # so any change to these declarations can silently break our build or
    # change runtime behavior. This subcommand fetches the upstream copies at
    # a target ref and diffs them against whatever is currently vendored
    # under thirdparty/llama.cpp/include/, so the surprise lands on the
    # developer running the bump rather than on CI after the tag is pushed.

    @option(
        "--llama-version",
        default=LLAMACPP_VERSION,
        help=f"llama.cpp ref to compare against (default: {LLAMACPP_VERSION}).",
    )
    @option(
        "--show-diff-lines",
        type=int,
        default=120,
        help="cap unified-diff output at this many lines (default: 120)",
    )
    def do_bump_check(self, args: argparse.Namespace) -> None:
        """compare vendored server-{context,http}.h against an upstream ref"""
        import difflib
        import re
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError, URLError

        ref = args.llama_version
        # All vendored upstream-internal headers chimera links against
        # or compiles against. The server-* pair were the only entries
        # historically; widening to common.h / arg.h / chat.h / mtmd.h /
        # llama.h surfaces drift in the larger public + common surface
        # chimera also pokes (common_params fields, llama_tokenize
        # signature, mtmd helper protos, etc.) without any extra
        # plumbing — same script, same vendored copies under
        # thirdparty/llama.cpp/include/, just more files compared.
        files = [
            "include/llama.h",
            "common/common.h",
            "common/arg.h",
            "common/chat.h",
            "tools/mtmd/mtmd.h",
            "tools/server/server-context.h",
            "tools/server/server-http.h",
        ]
        repo = "ggml-org/llama.cpp"
        vendored_dir = self.project.cwd / "thirdparty" / "llama.cpp" / "include"

        # Cheap "interesting symbol" extractor: catches struct/class/enum
        # declarations, handler_t members (the upstream split that broke
        # us last week was exactly a new handler_t field), and bare
        # function declarations. Not a real C++ parser; it surfaces the
        # surface-area changes a chimera dev needs to inspect.
        symbol_patterns = [
            re.compile(r"^\s*(?:struct|class|enum(?:\s+class)?)\s+(\w+)"),
            re.compile(r"^\s*handler_t\s+(\w+)\s*;"),
            re.compile(r"^\s*[\w:<>&*\s]+\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*[;{]"),
        ]

        def extract_symbols(text: str) -> set[str]:
            out: set[str] = set()
            for line in text.splitlines():
                for pat in symbol_patterns:
                    m = pat.match(line)
                    if m:
                        name = m.group(1)
                        if name and not name.startswith("_"):
                            out.add(name)
            return out

        def fetch_upstream(path: str) -> Optional[str]:
            url = f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"
            try:
                req = Request(url, headers={"User-Agent": "chimera-bump-check"})
                with urlopen(req, timeout=30) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except HTTPError as e:
                self.log.error(f"GET {url} -> HTTP {e.code}")
                return None
            except URLError as e:
                self.log.error(f"GET {url} -> {e}")
                return None

        if not vendored_dir.exists():
            self.log.error(
                f"No vendored headers found at {vendored_dir}. "
                f"Run `make build` (or `python scripts/manage.py build --llama-cpp`) first."
            )
            sys.exit(2)

        print(f"chimera bump-check: comparing vendored headers against {repo}@{ref}")
        print()

        total_changes = 0
        for upstream_path in files:
            basename = os.path.basename(upstream_path)
            vendored_path = vendored_dir / basename
            print(f"=== {basename} ===")
            if not vendored_path.exists():
                print(f"  (no vendored copy at {vendored_path}; skipped)\n")
                continue

            new_text = fetch_upstream(upstream_path)
            if new_text is None:
                print("  (failed to fetch upstream copy; skipped)\n")
                continue

            old_text = vendored_path.read_text(encoding="utf-8")
            if old_text == new_text:
                print("  no changes")
                print()
                continue

            old_syms = extract_symbols(old_text)
            new_syms = extract_symbols(new_text)
            removed = sorted(old_syms - new_syms)
            added = sorted(new_syms - old_syms)
            total_changes += 1

            if removed:
                print(f"  removed symbols ({len(removed)}):")
                for s in removed:
                    print(f"    - {s}")
            if added:
                print(f"  added symbols ({len(added)}):")
                for s in added:
                    print(f"    + {s}")
            if not removed and not added:
                print("  symbol set unchanged; signatures or comments differ")

            diff = list(
                difflib.unified_diff(
                    old_text.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=f"vendored/{basename}",
                    tofile=f"upstream@{ref}/{basename}",
                    n=2,
                )
            )
            cap = max(0, args.show_diff_lines)
            print(f"  unified diff ({len(diff)} lines, capped at {cap}):")
            for line in diff[:cap]:
                sys.stdout.write("    " + line)
            if len(diff) > cap:
                print(f"    ... ({len(diff) - cap} more lines elided)")
            print()

        if total_changes == 0:
            print("clean: vendored headers match upstream at this ref.")
            return
        print(
            f"{total_changes} header(s) changed. Audit:\n"
            f"  - chimera_serve.cpp bindings (handler_t fields)\n"
            f"  - src/chimera/CMakeLists.txt link order / archive groups\n"
            f"  - server-http.cpp source copy under thirdparty/llama.cpp/src-aux/\n"
            f"Then rerun `python scripts/manage.py build --llama-cpp --llama-version={ref}`."
        )
        sys.exit(2)

    # ------------------------------------------------------------------------
    # clean

    @opt("--reset", "-r", "also remove thirdparty/<dep>/{bin,lib,include}")
    @opt("--verbose", "-v", "verbose cleaning ops")
    def do_clean(self, args: argparse.Namespace) -> None:
        """remove build/ (and optionally thirdparty dep install dirs)"""
        verbose = args.verbose
        self.remove(self.project.cwd / "build", silent=not verbose)
        if args.reset:
            thirdparty = self.project.cwd / "thirdparty"
            for dep in [
                "llama.cpp",
                "whisper.cpp",
                "stable-diffusion.cpp",
                "linenoise",
            ]:
                dep_dir = thirdparty / dep
                for subdir in ["bin", "lib", "include"]:
                    self.remove(dep_dir / subdir, silent=not verbose)
        self.log.info("Clean complete")

    # ------------------------------------------------------------------------
    # download

    @opt("--llama", "-l", "download default Llama model")
    @opt("--whisper", "-w", "download Whisper model")
    @option(
        "--whisper-model",
        "-W",
        default="base.en",
        help="whisper model name (default: base.en)",
    )
    @option(
        "--models-dir",
        "-d",
        default="models",
        help="models directory (default: models)",
    )
    def do_download(self, args: argparse.Namespace) -> None:
        """download a test model for `chimera gen` / `chimera whisper`"""
        models_dir = Path(args.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        if args.llama:
            model_name = "Llama-3.2-1B-Instruct-Q8_0.gguf"
            model_path = models_dir / model_name
            if model_path.exists():
                self.log.info(f"Model already exists: {model_path}")
            else:
                url = (
                    f"https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/"
                    f"resolve/main/{model_name}"
                )
                self.log.info(f"Downloading {model_name}...")
                urlretrieve(url, model_path)
                self.log.info(f"Downloaded to {model_path}")

        if args.whisper:
            model_name = args.whisper_model
            valid_models = [
                "tiny",
                "tiny.en",
                "tiny-q5_1",
                "tiny.en-q5_1",
                "tiny-q8_0",
                "base",
                "base.en",
                "base-q5_1",
                "base.en-q5_1",
                "base-q8_0",
                "small",
                "small.en",
                "small.en-tdrz",
                "small-q5_1",
                "small.en-q5_1",
                "small-q8_0",
                "medium",
                "medium.en",
                "medium-q5_0",
                "medium.en-q5_0",
                "medium-q8_0",
                "large-v1",
                "large-v2",
                "large-v2-q5_0",
                "large-v2-q8_0",
                "large-v3",
                "large-v3-q5_0",
                "large-v3-turbo",
                "large-v3-turbo-q5_0",
                "large-v3-turbo-q8_0",
            ]
            if model_name not in valid_models:
                self.log.error(f"Invalid whisper model: {model_name}")
                self.log.info(f"Available models: {', '.join(valid_models)}")
                return

            model_file = f"ggml-{model_name}.bin"
            model_path = models_dir / model_file
            if model_path.exists():
                self.log.info(f"Model already exists: {model_path}")
            else:
                if "tdrz" in model_name:
                    src = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
                else:
                    src = "https://huggingface.co/ggerganov/whisper.cpp"
                url = f"{src}/resolve/main/ggml-{model_name}.bin"
                self.log.info(f"Downloading ggml-{model_name}.bin...")
                urlretrieve(url, model_path)
                self.log.info(f"Downloaded to {model_path}")

        if not args.llama and not args.whisper:
            self.log.info("Specify --llama or --whisper to download models")


if __name__ == "__main__":
    Application().cmdline()

cmake_minimum_required(VERSION 3.20)

# Extract chimera + bundled third-party versions from scripts/manage.py
# (the single source of truth). Falls back to "unknown" if the regex misses.
file(READ "${CMAKE_SOURCE_DIR}/scripts/manage.py" _manage_py)
foreach(_pair "CHIMERA_VERSION" "LLAMACPP_VERSION" "WHISPERCPP_VERSION" "SDCPP_VERSION" "LINENOISE_VERSION" "SQLITE_VERSION" "SQLITE_VEC_VERSION")
    string(REGEX MATCH "${_pair} = \"([^\"]+)\"" _ "${_manage_py}")
    if(CMAKE_MATCH_1)
        set(${_pair} "${CMAKE_MATCH_1}")
    else()
        set(${_pair} "unknown")
    endif()
endforeach()
unset(_manage_py)

project(chimera LANGUAGES C CXX VERSION "${CHIMERA_VERSION}")

message(STATUS "chimera version:        ${CHIMERA_VERSION}")
message(STATUS "llama.cpp version:      ${LLAMACPP_VERSION}")
message(STATUS "whisper.cpp version:    ${WHISPERCPP_VERSION}")
message(STATUS "sd.cpp version:         ${SDCPP_VERSION}")
message(STATUS "linenoise version:      ${LINENOISE_VERSION}")
message(STATUS "sqlite version:         ${SQLITE_VERSION}")
message(STATUS "sqlite-vec version:     ${SQLITE_VEC_VERSION}")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
# stable-diffusion.cpp must NOT use its own vendored ggml — it has to share
# llama.cpp's ggml so the final static link has a single backend set.
option(SD_USE_VENDORED_GGML "Link stable-diffusion against its own vendored ggml" OFF)

if(APPLE)
    set(_default_metal ON)
else()
    set(_default_metal OFF)
endif()

option(GGML_METAL   "Enable Metal backend (macOS)" ${_default_metal})
option(GGML_CUDA    "Enable CUDA backend"          OFF)
option(GGML_VULKAN  "Enable Vulkan backend"        OFF)
option(GGML_SYCL    "Enable SYCL backend"          OFF)
option(GGML_HIP     "Enable HIP/ROCm backend"      OFF)
option(GGML_OPENCL  "Enable OpenCL backend"        OFF)

# Environment-variable overrides (mirror scripts/manage.py conventions).
foreach(_var GGML_METAL GGML_CUDA GGML_VULKAN GGML_SYCL GGML_HIP GGML_OPENCL)
    if(DEFINED ENV{${_var}})
        if("$ENV{${_var}}" STREQUAL "1")
            set(${_var} ON  CACHE BOOL "" FORCE)
        elseif("$ENV{${_var}}" STREQUAL "0")
            set(${_var} OFF CACHE BOOL "" FORCE)
        endif()
    endif()
endforeach()

if(DEFINED ENV{SD_USE_VENDORED_GGML} AND "$ENV{SD_USE_VENDORED_GGML}" STREQUAL "1")
    set(SD_USE_VENDORED_GGML ON CACHE BOOL "" FORCE)
endif()

# -----------------------------------------------------------------------------
# Optional linenoise integration
# -----------------------------------------------------------------------------
# CHIMERA_LINENOISE:
#   ON   = require liblinenoise.a (fail configure if missing)
#   OFF  = do not link, even if the lib is present
#   AUTO = link if thirdparty/linenoise/lib/liblinenoise.a exists, else skip
# When linked, CHIMERA_HAS_LINENOISE is defined for the chimera target so
# command_chat can opt in to the readline-style input path.
set(CHIMERA_LINENOISE "AUTO" CACHE STRING "Link liblinenoise (ON / OFF / AUTO)")
set_property(CACHE CHIMERA_LINENOISE PROPERTY STRINGS "AUTO" "ON" "OFF")

set(LINENOISE_DIR     "${CMAKE_SOURCE_DIR}/thirdparty/linenoise")
set(LINENOISE_INCLUDE "${LINENOISE_DIR}/include")
set(LINENOISE_LIB     "${LINENOISE_DIR}/lib")

set(_linenoise_probe "${LINENOISE_LIB}/liblinenoise.a")
if(WIN32)
    set(_linenoise_probe "${LINENOISE_LIB}/linenoise.lib")
endif()

set(CHIMERA_LINK_LINENOISE OFF)
if("${CHIMERA_LINENOISE}" STREQUAL "ON")
    if(NOT EXISTS "${_linenoise_probe}")
        message(FATAL_ERROR
            "CHIMERA_LINENOISE=ON but ${_linenoise_probe} not found.\n"
            "Run: python scripts/manage.py build -L  (or --all)")
    endif()
    set(CHIMERA_LINK_LINENOISE ON)
elseif("${CHIMERA_LINENOISE}" STREQUAL "AUTO")
    if(EXISTS "${_linenoise_probe}")
        set(CHIMERA_LINK_LINENOISE ON)
    endif()
endif()
unset(_linenoise_probe)
message(STATUS "linenoise:              ${CHIMERA_LINENOISE} -> link=${CHIMERA_LINK_LINENOISE}")

# -----------------------------------------------------------------------------
# Optional embedded webui (experimental)
# -----------------------------------------------------------------------------
# CHIMERA_WEBUI_EMBED:
#   ON   = bake upstream llama.cpp's prebuilt web UI bundle into the binary
#          (xxd-encoded asset .hpp files included by server-http.cpp). Adds
#          ~7 MB to the binary; routes GET / + /bundle.{js,css} when the
#          server is up. Compiled-in UI is pinned to whichever llama.cpp
#          version chimera vendored, so the UI lifecycle is coupled to the
#          chimera release cadence.
#   OFF  = no webui routes; default. Binary stays lean.
#   AUTO = ON if thirdparty/llama.cpp/src-aux/webui/ exists, else OFF.
# Disable at runtime regardless of build choice with `chimera serve --no-webui`.
set(CHIMERA_WEBUI_EMBED "OFF" CACHE STRING "Embed llama.cpp web UI (ON / OFF / AUTO)")
set_property(CACHE CHIMERA_WEBUI_EMBED PROPERTY STRINGS "AUTO" "ON" "OFF")

set(_webui_probe "${CMAKE_SOURCE_DIR}/thirdparty/llama.cpp/src-aux/webui/index.html")
set(CHIMERA_LINK_WEBUI OFF)
if("${CHIMERA_WEBUI_EMBED}" STREQUAL "ON")
    if(NOT EXISTS "${_webui_probe}")
        message(FATAL_ERROR
            "CHIMERA_WEBUI_EMBED=ON but ${_webui_probe} not found.\n"
            "Run: python scripts/manage.py build --llama  (or --all)")
    endif()
    set(CHIMERA_LINK_WEBUI ON)
elseif("${CHIMERA_WEBUI_EMBED}" STREQUAL "AUTO")
    if(EXISTS "${_webui_probe}")
        set(CHIMERA_LINK_WEBUI ON)
    endif()
endif()
unset(_webui_probe)
message(STATUS "webui:                  ${CHIMERA_WEBUI_EMBED} -> link=${CHIMERA_LINK_WEBUI}")

# -----------------------------------------------------------------------------
# Optional modality libraries: whisper.cpp + stable-diffusion.cpp
# -----------------------------------------------------------------------------
# CHIMERA_WITH_WHISPER / CHIMERA_WITH_SD:
#   ON   = require the static lib (fail configure if missing)
#   OFF  = drop the modality completely. Without WHISPER the `whisper`
#          subcommand and `serve --enable-audio` flag + /v1/audio/* routes
#          disappear. Without SD the `sd` subcommand, `serve --enable-image`
#          flag, and /v1/images/* routes disappear. `gen --mmproj --image`
#          stays available regardless because it routes through libmtmd
#          (llama.cpp's vision pipeline), not chimera_sd.
#   AUTO = link if the lib is present, else skip.
# Defaults to AUTO so a stock `make build` keeps every modality wired up
# (libwhisper.a + libstable-diffusion.a are produced by manage.py's default
# `--all`). Toggle a modality off when you want a smaller binary, faster
# link, or zero dependency on whisper/SD upstream churn.
set(CHIMERA_WITH_WHISPER "AUTO" CACHE STRING "Link whisper.cpp (ON / OFF / AUTO)")
set_property(CACHE CHIMERA_WITH_WHISPER PROPERTY STRINGS "AUTO" "ON" "OFF")
set(CHIMERA_WITH_SD      "AUTO" CACHE STRING "Link stable-diffusion.cpp (ON / OFF / AUTO)")
set_property(CACHE CHIMERA_WITH_SD      PROPERTY STRINGS "AUTO" "ON" "OFF")

set(_whisper_probe "${CMAKE_SOURCE_DIR}/thirdparty/whisper.cpp/lib/libwhisper.a")
if(WIN32)
    set(_whisper_probe "${CMAKE_SOURCE_DIR}/thirdparty/whisper.cpp/lib/whisper.lib")
endif()
set(CHIMERA_LINK_WHISPER OFF)
if("${CHIMERA_WITH_WHISPER}" STREQUAL "ON")
    if(NOT EXISTS "${_whisper_probe}")
        message(FATAL_ERROR
            "CHIMERA_WITH_WHISPER=ON but ${_whisper_probe} not found.\n"
            "Run: python scripts/manage.py build -w  (or --all)")
    endif()
    set(CHIMERA_LINK_WHISPER ON)
elseif("${CHIMERA_WITH_WHISPER}" STREQUAL "AUTO")
    if(EXISTS "${_whisper_probe}")
        set(CHIMERA_LINK_WHISPER ON)
    endif()
endif()
unset(_whisper_probe)
message(STATUS "whisper.cpp:            ${CHIMERA_WITH_WHISPER} -> link=${CHIMERA_LINK_WHISPER}")

set(_sd_probe "${CMAKE_SOURCE_DIR}/thirdparty/stable-diffusion.cpp/lib/libstable-diffusion.a")
if(WIN32)
    set(_sd_probe "${CMAKE_SOURCE_DIR}/thirdparty/stable-diffusion.cpp/lib/stable-diffusion.lib")
endif()
set(CHIMERA_LINK_SD OFF)
if("${CHIMERA_WITH_SD}" STREQUAL "ON")
    if(NOT EXISTS "${_sd_probe}")
        message(FATAL_ERROR
            "CHIMERA_WITH_SD=ON but ${_sd_probe} not found.\n"
            "Run: python scripts/manage.py build -s  (or --all)")
    endif()
    set(CHIMERA_LINK_SD ON)
elseif("${CHIMERA_WITH_SD}" STREQUAL "AUTO")
    if(EXISTS "${_sd_probe}")
        set(CHIMERA_LINK_SD ON)
    endif()
endif()
unset(_sd_probe)
message(STATUS "stable-diffusion.cpp:   ${CHIMERA_WITH_SD} -> link=${CHIMERA_LINK_SD}")

# stable-diffusion.cpp requires GGML_MAX_NAME=128 for long tensor names.
# When SD shares llama.cpp's ggml, both must use the same value. Only
# needed when SD is actually linked in.
if(CHIMERA_LINK_SD AND NOT SD_USE_VENDORED_GGML)
    add_definitions(-DGGML_MAX_NAME=128)
endif()

message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "GGML_METAL:  ${GGML_METAL}")
message(STATUS "GGML_CUDA:   ${GGML_CUDA}")
message(STATUS "GGML_VULKAN: ${GGML_VULKAN}")
message(STATUS "GGML_HIP:    ${GGML_HIP}")
message(STATUS "GGML_SYCL:   ${GGML_SYCL}")
message(STATUS "GGML_OPENCL: ${GGML_OPENCL}")

# -----------------------------------------------------------------------------
# Build-time intent capture for `chimera info`.
#
# We bake the enabled-backend list + tuning knobs into compile-time string
# macros so `chimera info` can report what the binary was *built* for in
# addition to what its ggml registry happens to expose at runtime. The two
# normally agree; when they don't (driver missing, registry init failed,
# etc.) the divergence is the bug-report signal.
#
# Modelled on cyllama's build_config.json but without the sidecar file —
# `#define` keeps chimera's "one static binary, nothing else to install"
# posture intact.
# -----------------------------------------------------------------------------
set(_CHIMERA_BUILT_BACKENDS "")
foreach(_b "GGML_METAL=Metal" "GGML_CUDA=CUDA" "GGML_VULKAN=Vulkan"
           "GGML_HIP=HIP" "GGML_SYCL=SYCL" "GGML_OPENCL=OpenCL"
           "GGML_BLAS=BLAS")
    string(REPLACE "=" ";" _kv "${_b}")
    list(GET _kv 0 _flag)
    list(GET _kv 1 _label)
    if(${_flag})
        if(_CHIMERA_BUILT_BACKENDS STREQUAL "")
            set(_CHIMERA_BUILT_BACKENDS "${_label}")
        else()
            set(_CHIMERA_BUILT_BACKENDS "${_CHIMERA_BUILT_BACKENDS},${_label}")
        endif()
    endif()
endforeach()
if(_CHIMERA_BUILT_BACKENDS STREQUAL "")
    set(_CHIMERA_BUILT_BACKENDS "CPU")
endif()
set(CHIMERA_BUILT_BACKENDS "${_CHIMERA_BUILT_BACKENDS}")

# Tuning knobs we surface in `info`. Each becomes an empty-string macro
# when not set, so the consumer just filters out the empties.
set(CHIMERA_CUDA_ARCHITECTURES "$ENV{CMAKE_CUDA_ARCHITECTURES}")
set(CHIMERA_HIP_ARCHITECTURES  "$ENV{CMAKE_HIP_ARCHITECTURES}")
set(CHIMERA_BLAS_VENDOR        "$ENV{GGML_BLAS_VENDOR}")
set(CHIMERA_CUDA_FORCE_MMQ     "$ENV{GGML_CUDA_FORCE_MMQ}")
set(CHIMERA_CUDA_FORCE_CUBLAS  "$ENV{GGML_CUDA_FORCE_CUBLAS}")
set(CHIMERA_HIP_ROCWMMA_FATTN  "$ENV{GGML_HIP_ROCWMMA_FATTN}")

message(STATUS "built backends:         ${CHIMERA_BUILT_BACKENDS}")

# -----------------------------------------------------------------------------
# Platform-specific static-library naming
# -----------------------------------------------------------------------------
if(APPLE)
    set(STATIC_LIB_PREFIX "lib")
    set(STATIC_LIB_SUFFIX ".a")
elseif(WIN32)
    set(STATIC_LIB_PREFIX "")
    set(STATIC_LIB_SUFFIX ".lib")
else()
    set(STATIC_LIB_PREFIX "lib")
    set(STATIC_LIB_SUFFIX ".a")
endif()

# -----------------------------------------------------------------------------
# Thirdparty paths
# -----------------------------------------------------------------------------
# scripts/manage.py builds llama.cpp / whisper.cpp / stable-diffusion.cpp into
# thirdparty/<project>/{include,lib}. Run it once before invoking cmake:
#
#   python scripts/manage.py build --all --deps-only --sd-shared-ggml
#
set(LLAMACPP_DIR    "${CMAKE_SOURCE_DIR}/thirdparty/llama.cpp")
set(WHISPERCPP_DIR  "${CMAKE_SOURCE_DIR}/thirdparty/whisper.cpp")
set(SDCPP_DIR       "${CMAKE_SOURCE_DIR}/thirdparty/stable-diffusion.cpp")

set(LLAMACPP_INCLUDE   "${LLAMACPP_DIR}/include")
set(WHISPERCPP_INCLUDE "${WHISPERCPP_DIR}/include")
set(SDCPP_INCLUDE      "${SDCPP_DIR}/include")

set(LLAMACPP_LIB    "${LLAMACPP_DIR}/lib")
set(WHISPERCPP_LIB  "${WHISPERCPP_DIR}/lib")
set(SDCPP_LIB       "${SDCPP_DIR}/lib")

foreach(_dir "${LLAMACPP_LIB}" "${WHISPERCPP_LIB}" "${SDCPP_LIB}")
    if(NOT EXISTS "${_dir}")
        message(FATAL_ERROR
            "Missing build artifact directory: ${_dir}\n"
            "Run: python scripts/manage.py build --all --deps-only --sd-shared-ggml")
    endif()
endforeach()

function(static_lib result lib_dir name)
    set(${result} "${lib_dir}/${STATIC_LIB_PREFIX}${name}${STATIC_LIB_SUFFIX}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Static libraries
# -----------------------------------------------------------------------------
static_lib(LIB_LLAMA           "${LLAMACPP_LIB}" "llama")
static_lib(LIB_LLAMA_COMMON    "${LLAMACPP_LIB}" "llama-common")
static_lib(LIB_GGML            "${LLAMACPP_LIB}" "ggml")
static_lib(LIB_GGML_BASE       "${LLAMACPP_LIB}" "ggml-base")
static_lib(LIB_GGML_CPU        "${LLAMACPP_LIB}" "ggml-cpu")
static_lib(LIB_MTMD            "${LLAMACPP_LIB}" "mtmd")
static_lib(LIB_SERVER_CONTEXT  "${LLAMACPP_LIB}" "server-context")
static_lib(LIB_CPP_HTTPLIB     "${LLAMACPP_LIB}" "cpp-httplib")

# server-http.cpp is not part of libserver-context.a (upstream compiles it
# directly into the llama-server executable). manage.py drops it here so the
# chimera target can pull it into its own compilation.
set(LLAMACPP_SRC_AUX "${LLAMACPP_DIR}/src-aux")

# SQLite + sqlite-vec are shipped as single-translation-unit amalgamations
# (sqlite3.c, sqlite-vec.c). manage.py vendors them under thirdparty/<name>/
# with the same include/+src-aux/ layout as the llama-server pieces above.
# chimera's CMakeLists pulls both .c files into the chimera target's source
# list rather than building a separate library.
set(SQLITE_DIR     "${CMAKE_SOURCE_DIR}/thirdparty/sqlite")
set(SQLITE_VEC_DIR "${CMAKE_SOURCE_DIR}/thirdparty/sqlite-vec")
set(SQLITE_INCLUDE     "${SQLITE_DIR}/include")
set(SQLITE_VEC_INCLUDE "${SQLITE_VEC_DIR}/include")
set(SQLITE_SRC_AUX     "${SQLITE_DIR}/src-aux")
set(SQLITE_VEC_SRC_AUX "${SQLITE_VEC_DIR}/src-aux")
foreach(_aux "${SQLITE_SRC_AUX}/sqlite3.c" "${SQLITE_VEC_SRC_AUX}/sqlite-vec.c")
    if(NOT EXISTS "${_aux}")
        message(FATAL_ERROR
            "Missing vendored source: ${_aux}\n"
            "Run: python scripts/manage.py build --sqlite --sqlite-vec  (or --all)")
    endif()
endforeach()

if(GGML_METAL)
    static_lib(LIB_GGML_BLAS  "${LLAMACPP_LIB}" "ggml-blas")
    static_lib(LIB_GGML_METAL "${LLAMACPP_LIB}" "ggml-metal")
endif()
if(GGML_CUDA)
    static_lib(LIB_GGML_CUDA   "${LLAMACPP_LIB}" "ggml-cuda")
endif()
if(GGML_VULKAN)
    static_lib(LIB_GGML_VULKAN "${LLAMACPP_LIB}" "ggml-vulkan")
endif()
if(GGML_SYCL)
    static_lib(LIB_GGML_SYCL   "${LLAMACPP_LIB}" "ggml-sycl")
endif()
if(GGML_HIP)
    static_lib(LIB_GGML_HIP    "${LLAMACPP_LIB}" "ggml-hip")
endif()
if(GGML_OPENCL)
    static_lib(LIB_GGML_OPENCL "${LLAMACPP_LIB}" "ggml-opencl")
endif()

static_lib(LIB_WHISPER_COMMON "${WHISPERCPP_LIB}" "common")
static_lib(LIB_WHISPER        "${WHISPERCPP_LIB}" "whisper")
static_lib(LIB_SD             "${SDCPP_LIB}"      "stable-diffusion")
if(CHIMERA_LINK_LINENOISE)
    static_lib(LIB_LINENOISE  "${LINENOISE_LIB}"  "linenoise")
endif()

# -----------------------------------------------------------------------------
# System libraries
# -----------------------------------------------------------------------------
set(SYSTEM_LIBS "")

if(NOT WIN32)
    list(APPEND SYSTEM_LIBS pthread)
else()
    list(APPEND SYSTEM_LIBS kernel32 user32 advapi32)
endif()

# cpp-httplib on macOS reads trust-anchor certs from the system keychain
# (SecCertificateCopyData / SecTrustCopyAnchorCertificates). Link Security
# + CoreFoundation so those symbols resolve. Required for `chimera serve`.
if(APPLE)
    list(APPEND SYSTEM_LIBS "-framework Security" "-framework CoreFoundation")
endif()

# OpenMP: ggml-cpu's CMake enables OpenMP by default when the compiler
# supports it. On Linux + gcc that means libgomp symbols (GOMP_barrier,
# GOMP_parallel, omp_get_thread_num, ...) end up in libggml-cpu.a, and
# the final chimera link needs the runtime. macOS / Apple clang has no
# OpenMP by default so find_package fails silently and ggml builds
# without OpenMP -- nothing to link.
# OpenSSL: cpp-httplib is compiled with LLAMA_OPENSSL=ON (manage.py) so it
# references X509_*, SSL_*, EVP_*, etc. The chimera serve subcommand uses
# cpp-httplib to host the OpenAI-compatible API; we therefore need the
# system OpenSSL runtime at link time. Brew-installed OpenSSL on macOS
# typically lives outside CMake's default search paths -- caller can set
# OPENSSL_ROOT_DIR if find_package can't locate it.
find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL:                found (${OPENSSL_VERSION})")
list(APPEND SYSTEM_LIBS OpenSSL::SSL OpenSSL::Crypto)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP:                 found (${OpenMP_CXX_VERSION})")
    list(APPEND SYSTEM_LIBS OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP:                 not found (ok on macOS / Apple clang)")
endif()

if(GGML_METAL)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(METAL_FRAMEWORK      Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK   MetalKit   REQUIRED)
    list(APPEND SYSTEM_LIBS
        ${ACCELERATE_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
    )
endif()

if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SYSTEM_LIBS CUDA::cudart CUDA::cublas CUDA::cuda_driver)
endif()

if(GGML_VULKAN)
    find_package(Vulkan REQUIRED)
    list(APPEND SYSTEM_LIBS Vulkan::Vulkan)
endif()

if(GGML_HIP)
    find_package(hip      REQUIRED)
    find_package(hipblas  REQUIRED)
    find_package(rocblas  REQUIRED)
    list(APPEND SYSTEM_LIBS hip::host roc::rocblas roc::hipblas)
endif()

# -----------------------------------------------------------------------------
# Compile options
# -----------------------------------------------------------------------------
if(WIN32)
    set(CXX_COMPILE_OPTIONS /std:c++17 /EHsc /MD)
else()
    set(CXX_COMPILE_OPTIONS -std=c++17 -fvisibility=hidden -fvisibility-inlines-hidden)
endif()

# -----------------------------------------------------------------------------
# chimera target (shared source layout with cyllama)
# -----------------------------------------------------------------------------
set(CHIMERA_STB_IMPL "${CMAKE_SOURCE_DIR}/src/chimera/stb_impl.cpp")

add_subdirectory(src/chimera)

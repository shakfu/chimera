# bindings/CMakeLists.txt - builds the `chimera` Python extension module
# (nanobind) against chimera's three prebuilt static archives. Mirrors the
# link contract exercised by tests/external/ (the canonical external
# consumer) but produces a Python module instead of an executable.
#
# Prerequisites (the archives must already exist):
#   make build && python3 scripts/combine_archives.py
#
# Standalone build (for iterating on the bindings only):
#   cmake -S bindings -B bindings/build -DCHIMERA_BUILD_ROOT=$PWD/build
#   cmake --build bindings/build
# Then `import chimera` with bindings/build on PYTHONPATH.
#
# Packaged build: driven by bindings/pyproject.toml via scikit-build-core
#   (cd bindings && pip install .).

cmake_minimum_required(VERSION 3.18)
project(chimera_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- locate the prebuilt chimera archives ----------------------------------
if(NOT DEFINED CHIMERA_BUILD_ROOT)
    set(CHIMERA_BUILD_ROOT "${CMAKE_CURRENT_LIST_DIR}/../build" CACHE PATH
        "Chimera build dir containing libchimera*.a")
endif()
get_filename_component(CHIMERA_BUILD_ROOT "${CHIMERA_BUILD_ROOT}" ABSOLUTE)
set(_repo_root "${CMAKE_CURRENT_LIST_DIR}/..")

set(_libchimera            "${CHIMERA_BUILD_ROOT}/libchimera.a")
set(_libchimera_thirdparty "${CHIMERA_BUILD_ROOT}/libchimera_thirdparty.a")
set(_libchimera_ggml       "${CHIMERA_BUILD_ROOT}/libchimera_ggml.a")
foreach(_p IN ITEMS "${_libchimera}" "${_libchimera_thirdparty}" "${_libchimera_ggml}")
    if(NOT EXISTS "${_p}")
        message(FATAL_ERROR "missing archive: ${_p}\n"
                "Run `make build && python3 scripts/combine_archives.py` first.")
    endif()
endforeach()

# ---- Python + nanobind ------------------------------------------------------
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
# Locate nanobind's CMake config via the installed package (pip install nanobind).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(chimera STABLE_ABI chimera_ext.cpp)

# ---- headers (same source layout the external test assumes) -----------------
target_include_directories(chimera PRIVATE
    "${_repo_root}/src/chimera"
    "${_repo_root}/thirdparty"
    "${_repo_root}/build/llama.cpp/include"
    "${_repo_root}/build/llama.cpp/ggml/include"
    # chimera.hpp pulls common.h / chat.h / sampling.h / mtmd.h, which live
    # in the staged llama.cpp install root rather than build/llama.cpp/include.
    "${_repo_root}/thirdparty/llama.cpp/include"
)

# ---- modality + version defines (must match the libchimera.a build) ---------
# Default both modalities ON; override at configure time if your archives
# dropped one (e.g. -DCHIMERA_HAS_SD=OFF).
option(CHIMERA_HAS_WHISPER "Match the libchimera.a whisper-enabled build" ON)
option(CHIMERA_HAS_SD      "Match the libchimera.a sd-enabled build"      ON)
if(CHIMERA_HAS_WHISPER)
    target_compile_definitions(chimera PRIVATE CHIMERA_HAS_WHISPER)
endif()
if(CHIMERA_HAS_SD)
    target_compile_definitions(chimera PRIVATE CHIMERA_HAS_SD)
endif()

# CHIMERA_VERSION from the single source of truth (we bypass chimera_lib's
# PUBLIC compile defines by linking the raw archive).
file(STRINGS "${_repo_root}/scripts/manage.py" _ver_line
     REGEX "^CHIMERA_VERSION[ ]*=[ ]*\"")
if(_ver_line MATCHES "\"([^\"]+)\"")
    set(_chimera_version "${CMAKE_MATCH_1}")
else()
    set(_chimera_version "unknown")
endif()
target_compile_definitions(chimera PRIVATE "CHIMERA_VERSION=\"${_chimera_version}\"")

# ---- the link contract ------------------------------------------------------
#   libchimera.a            : normal link
#   libchimera_thirdparty.a : normal link
#   libchimera_ggml.a       : WHOLE-ARCHIVE (ggml backend static initializers
#                             must run or GPU backends silently don't register)
#
# OpenSSL: only needed when the archives were built with CHIMERA_OPENSSL=ON
# (default is OFF -> vendored SHA-256, no OpenSSL). Link it only if present so
# the default OpenSSL-free build doesn't require it installed.
find_package(OpenSSL QUIET)
find_package(OpenMP QUIET)

if(APPLE)
    target_link_libraries(chimera PRIVATE
        "${_libchimera}"
        "${_libchimera_thirdparty}"
        "-Wl,-force_load,${_libchimera_ggml}"
        "-framework Security"
        "-framework CoreFoundation"
        "-framework Accelerate"
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
        pthread
    )
    if(OpenSSL_FOUND)
        target_link_libraries(chimera PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    endif()
elseif(UNIX)
    target_link_libraries(chimera PRIVATE
        "${_libchimera}"
        "${_libchimera_thirdparty}"
        "-Wl,--whole-archive" "${_libchimera_ggml}" "-Wl,--no-whole-archive"
        pthread dl m
    )
    if(OpenSSL_FOUND)
        target_link_libraries(chimera PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(chimera PRIVATE OpenMP::OpenMP_CXX)
    endif()
elseif(WIN32)
    get_filename_component(_ggml_name "${_libchimera_ggml}" NAME)
    target_link_libraries(chimera PRIVATE
        "${_libchimera}" "${_libchimera_thirdparty}" "${_libchimera_ggml}")
    target_link_options(chimera PRIVATE "/WHOLEARCHIVE:${_ggml_name}")
endif()

# ---- install (scikit-build-core places this into the wheel) -----------------
install(TARGETS chimera LIBRARY DESTINATION .)

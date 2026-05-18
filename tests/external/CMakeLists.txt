# tests/external/CMakeLists.txt - standalone smoke test that links the
# two chimera static archives (libchimera.a + libchimera_deps.a) as an
# external consumer would. Deliberately does NOT use the chimera_lib
# target from the parent build - the point is to exercise the raw
# archive contract a non-CMake / packaged consumer would face.
#
# Configure:
#   cmake -S tests/external -B tests/external/build \
#         -DCHIMERA_BUILD_ROOT=/abs/path/to/chimera/build
# Build:
#   cmake --build tests/external/build
# Run:
#   tests/external/build/chimera_smoke
#
# CHIMERA_BUILD_ROOT defaults to <repo>/build assuming the standard
# layout. Override to point at a different build dir.

cmake_minimum_required(VERSION 3.18)
project(chimera_external_smoke CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CHIMERA_BUILD_ROOT)
    set(CHIMERA_BUILD_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../build" CACHE PATH
        "Chimera build dir containing libchimera.a + libchimera_deps.a")
endif()
get_filename_component(CHIMERA_BUILD_ROOT "${CHIMERA_BUILD_ROOT}" ABSOLUTE)

set(_libchimera            "${CHIMERA_BUILD_ROOT}/libchimera.a")
set(_libchimera_thirdparty "${CHIMERA_BUILD_ROOT}/libchimera_thirdparty.a")
set(_libchimera_ggml       "${CHIMERA_BUILD_ROOT}/libchimera_ggml.a")
foreach(_p IN ITEMS "${_libchimera}" "${_libchimera_thirdparty}" "${_libchimera_ggml}")
    if(NOT EXISTS "${_p}")
        message(FATAL_ERROR "missing archive: ${_p}\n"
                "Run `make build && python3 scripts/combine_archives.py` first.")
    endif()
endforeach()

# Headers. We assume the same source layout as the chimera repo. A
# packaged consumer would receive these as an install/include/ tree;
# pointing at the in-repo locations is the simplest way to smoke-test
# the archives without standing up a real install step.
set(_repo_root "${CMAKE_CURRENT_LIST_DIR}/../..")
set(_llama_inc "${_repo_root}/build/llama.cpp/include"
               "${_repo_root}/build/llama.cpp/ggml/include")

add_executable(chimera_smoke smoke.cpp)
target_include_directories(chimera_smoke PRIVATE
    "${_repo_root}/src/chimera"
    "${_repo_root}/thirdparty"
    ${_llama_inc}
)

# Forward the same CHIMERA_HAS_* defines libchimera.a was compiled
# with so the conditional sections in smoke.cpp match. Default both
# ON; override at configure time if your build dropped a modality.
option(CHIMERA_HAS_WHISPER "Match the libchimera.a whisper-enabled build" ON)
option(CHIMERA_HAS_SD      "Match the libchimera.a sd-enabled build"      ON)
if(CHIMERA_HAS_WHISPER)
    target_compile_definitions(chimera_smoke PRIVATE CHIMERA_HAS_WHISPER)
endif()
if(CHIMERA_HAS_SD)
    target_compile_definitions(chimera_smoke PRIVATE CHIMERA_HAS_SD)
endif()

# Pull CHIMERA_VERSION out of scripts/manage.py (the project's single
# source of truth, parsed in the same way the top-level CMakeLists
# does). Without this, chimera.h's "unknown" fallback would kick in
# because we bypass chimera_lib's PUBLIC compile defines.
file(STRINGS "${_repo_root}/scripts/manage.py" _ver_line
     REGEX "^CHIMERA_VERSION[ ]*=[ ]*\"")
if(_ver_line MATCHES "\"([^\"]+)\"")
    set(_chimera_version "${CMAKE_MATCH_1}")
else()
    set(_chimera_version "unknown")
endif()
target_compile_definitions(chimera_smoke PRIVATE
    "CHIMERA_VERSION=\"${_chimera_version}\"")

# The link contract:
#   - libchimera.a              : normal link (chimera's own code)
#   - libchimera_thirdparty.a   : normal link (deps with no static-init
#                                 registration; whole-archiving would
#                                 pull in duplicate trim() etc. from
#                                 two upstream TUs)
#   - libchimera_ggml.a         : whole-archive REQUIRED so ggml
#                                 backend static initializers run
#
# Platform wrappers for the ggml archive:
#   macOS  : -Wl,-force_load,<path>
#   Linux  : -Wl,--whole-archive <path> -Wl,--no-whole-archive
#   Windows: /WHOLEARCHIVE:<basename>
#
# System libs chimera_lib's PUBLIC link list would normally carry.
# We bypass that target so list them explicitly. Matches the
# SYSTEM_LIBS composition in the top-level CMakeLists.txt.
find_package(OpenSSL REQUIRED)
find_package(OpenMP QUIET)

if(APPLE)
    target_link_libraries(chimera_smoke PRIVATE
        "${_libchimera}"
        "${_libchimera_thirdparty}"
        "-Wl,-force_load,${_libchimera_ggml}"
        OpenSSL::SSL OpenSSL::Crypto
        "-framework Security"
        "-framework CoreFoundation"
        "-framework Accelerate"
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
        pthread
    )
elseif(UNIX)
    target_link_libraries(chimera_smoke PRIVATE
        "${_libchimera}"
        "${_libchimera_thirdparty}"
        "-Wl,--whole-archive" "${_libchimera_ggml}" "-Wl,--no-whole-archive"
        OpenSSL::SSL OpenSSL::Crypto
        pthread dl m
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(chimera_smoke PRIVATE OpenMP::OpenMP_CXX)
    endif()
elseif(WIN32)
    get_filename_component(_ggml_name "${_libchimera_ggml}" NAME)
    target_link_libraries(chimera_smoke PRIVATE
        "${_libchimera}"
        "${_libchimera_thirdparty}"
        "${_libchimera_ggml}"
    )
    target_link_options(chimera_smoke PRIVATE "/WHOLEARCHIVE:${_ggml_name}")
endif()

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
               "${_repo_root}/build/llama.cpp/ggml/include"
               # The OOP header (chimera.hpp) re-includes chimera_llama.h,
               # which pulls common.h / chat.h / sampling.h / mtmd.h. Those
               # live in the staged llama.cpp install root, not in the
               # narrower build/llama.cpp/include dir that exposes only
               # llama.h. smoke.cpp doesn't need this; hpp_smoke.cpp does.
               "${_repo_root}/thirdparty/llama.cpp/include")

add_executable(chimera_smoke smoke.cpp)
add_executable(chimera_hpp_smoke hpp_smoke.cpp)
foreach(_t chimera_smoke chimera_hpp_smoke)
    target_include_directories(${_t} PRIVATE
        "${_repo_root}/src/chimera"
        "${_repo_root}/thirdparty"
        ${_llama_inc}
    )
endforeach()

# Forward the same CHIMERA_HAS_* defines libchimera.a was compiled
# with so the conditional sections in smoke.cpp match. Default both
# ON; override at configure time if your build dropped a modality.
option(CHIMERA_HAS_WHISPER "Match the libchimera.a whisper-enabled build" ON)
option(CHIMERA_HAS_SD      "Match the libchimera.a sd-enabled build"      ON)
foreach(_t chimera_smoke chimera_hpp_smoke)
    if(CHIMERA_HAS_WHISPER)
        target_compile_definitions(${_t} PRIVATE CHIMERA_HAS_WHISPER)
    endif()
    if(CHIMERA_HAS_SD)
        target_compile_definitions(${_t} PRIVATE CHIMERA_HAS_SD)
    endif()
endforeach()

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
foreach(_t chimera_smoke chimera_hpp_smoke)
    target_compile_definitions(${_t} PRIVATE
        "CHIMERA_VERSION=\"${_chimera_version}\"")
endforeach()

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

foreach(_t chimera_smoke chimera_hpp_smoke)
    if(APPLE)
        target_link_libraries(${_t} PRIVATE
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
        target_link_libraries(${_t} PRIVATE
            "${_libchimera}"
            "${_libchimera_thirdparty}"
            "-Wl,--whole-archive" "${_libchimera_ggml}" "-Wl,--no-whole-archive"
            OpenSSL::SSL OpenSSL::Crypto
            pthread dl m
        )
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${_t} PRIVATE OpenMP::OpenMP_CXX)
        endif()
    elseif(WIN32)
        get_filename_component(_ggml_name "${_libchimera_ggml}" NAME)
        target_link_libraries(${_t} PRIVATE
            "${_libchimera}"
            "${_libchimera_thirdparty}"
            "${_libchimera_ggml}"
        )
        target_link_options(${_t} PRIVATE "/WHOLEARCHIVE:${_ggml_name}")
    endif()
endforeach()

# -----------------------------------------------------------------------------
# CTest registration.
#
# Two test entries -- one per smoke binary. Labels let `ctest -L OOP` or
# `ctest -L PROCEDURAL` pick a single lane; `ctest -L MODEL_GATED` selects
# only the entries whose inference probes need a model file.
#
# Skip semantics:
#   The binaries print "SKIP (set CHIMERA_SMOKE_..." when their env-var
#   gated inference probes are disabled. SKIP_REGULAR_EXPRESSION maps that
#   pattern to ctest's "Skipped" outcome instead of "Passed" so a CI
#   dashboard can distinguish "ran the probe and it passed" from "the
#   probe was unavailable because no fixture model was set".
#
# Env-var inputs (optional, both binaries respect them):
#   CHIMERA_SMOKE_MODEL           - llama-side GGUF used by both binaries
#                                   to exercise inference paths.
#   CHIMERA_SMOKE_WHISPER_MODEL   - whisper GGML; chimera_hpp_smoke only.
#   CHIMERA_SMOKE_WHISPER_INPUT   - WAV file; chimera_hpp_smoke only.
# -----------------------------------------------------------------------------
enable_testing()

# Run tests from the repo root so the CHIMERA_SMOKE_MODEL /
# CHIMERA_SMOKE_WHISPER_* env vars can use repo-relative paths that
# match the shell invocations users will copy/paste from the docs.
set(_chimera_test_wd "${CMAKE_CURRENT_LIST_DIR}/../..")

add_test(
    NAME chimera_smoke
    COMMAND chimera_smoke
    WORKING_DIRECTORY "${_chimera_test_wd}"
)
set_tests_properties(chimera_smoke PROPERTIES
    LABELS "EXTERNAL;PROCEDURAL;MODEL_GATED"
    # Map the binary's "SKIP (set CHIMERA_SMOKE_MODEL ...)" line to
    # ctest's Skipped outcome so a CI dashboard can distinguish "ran
    # the probe and it passed" from "the probe was unavailable
    # because no fixture model was set". The link-contract probe
    # (ggml backend registration, llama symbol resolution) always
    # runs regardless of env vars.
    SKIP_REGULAR_EXPRESSION "inference probe: SKIP \\(set CHIMERA_SMOKE_MODEL"
    PASS_REGULAR_EXPRESSION "(^|\n)PASS"
    TIMEOUT 120
)

add_test(
    NAME chimera_hpp_smoke
    COMMAND chimera_hpp_smoke
    WORKING_DIRECTORY "${_chimera_test_wd}"
)
set_tests_properties(chimera_hpp_smoke PROPERTIES
    LABELS "EXTERNAL;OOP;MODEL_GATED"
    SKIP_REGULAR_EXPRESSION "inference probe: SKIP \\(set CHIMERA_SMOKE_MODEL"
    PASS_REGULAR_EXPRESSION "(^|\n)PASS"
    TIMEOUT 180
)

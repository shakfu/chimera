cmake_minimum_required(VERSION 3.20)

# Extract chimera + bundled third-party versions from scripts/manage.py
# (the single source of truth). Falls back to "unknown" if the regex misses.
file(READ "${CMAKE_SOURCE_DIR}/scripts/manage.py" _manage_py)
foreach(_pair "CHIMERA_VERSION" "LLAMACPP_VERSION" "WHISPERCPP_VERSION" "SDCPP_VERSION")
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

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
# WITH_DYLIB is always OFF for chimera (static-only target). It exists as a
# variable so the shared src/chimera/CMakeLists.txt slice can early-return.
set(WITH_DYLIB OFF)

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

# stable-diffusion.cpp requires GGML_MAX_NAME=128 for long tensor names.
# When SD shares llama.cpp's ggml, both must use the same value.
if(NOT SD_USE_VENDORED_GGML)
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
static_lib(LIB_LLAMA         "${LLAMACPP_LIB}" "llama")
static_lib(LIB_LLAMA_COMMON  "${LLAMACPP_LIB}" "llama-common")
static_lib(LIB_GGML          "${LLAMACPP_LIB}" "ggml")
static_lib(LIB_GGML_BASE     "${LLAMACPP_LIB}" "ggml-base")
static_lib(LIB_GGML_CPU      "${LLAMACPP_LIB}" "ggml-cpu")
static_lib(LIB_MTMD          "${LLAMACPP_LIB}" "mtmd")

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

# -----------------------------------------------------------------------------
# System libraries
# -----------------------------------------------------------------------------
set(SYSTEM_LIBS "")

if(NOT WIN32)
    list(APPEND SYSTEM_LIBS pthread)
else()
    list(APPEND SYSTEM_LIBS kernel32 user32 advapi32)
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

# -----------------------------------------------------------------------------
# chimera - statically linked llama.cpp / whisper.cpp / stable-diffusion.cpp
# multitool executable.
#
# Inputs (must be defined by the including CMakeLists.txt):
#   WITH_DYLIB, SD_USE_VENDORED_GGML  - build mode toggles
#   LLAMACPP_INCLUDE, WHISPERCPP_INCLUDE, SDCPP_INCLUDE
#   LIB_LLAMA, LIB_LLAMA_COMMON, LIB_MTMD
#   LIB_WHISPER, LIB_WHISPER_COMMON
#   LIB_SD
#   LIB_GGML, LIB_GGML_BASE, LIB_GGML_CPU
#   LIB_GGML_BLAS, LIB_GGML_METAL, LIB_GGML_CUDA, LIB_GGML_VULKAN,
#   LIB_GGML_SYCL,  LIB_GGML_HIP,   LIB_GGML_OPENCL   (only those enabled)
#   GGML_METAL, GGML_CUDA, GGML_VULKAN, GGML_SYCL, GGML_HIP, GGML_OPENCL
#   SYSTEM_LIBS, CXX_COMPILE_OPTIONS
#
# Source paths assume this file lives at <repo>/src/chimera/CMakeLists.txt.
# stb_impl.cpp is sourced from cyllama; standalone builds should ship their
# own copy and override CHIMERA_STB_IMPL before including this file.
# -----------------------------------------------------------------------------

if(WITH_DYLIB)
    return()
endif()

if(SD_USE_VENDORED_GGML)
    message(STATUS "chimera disabled: requires SD_USE_VENDORED_GGML=OFF")
    return()
endif()

if(NOT DEFINED CHIMERA_STB_IMPL)
    set(CHIMERA_STB_IMPL "${CMAKE_SOURCE_DIR}/src/cyllama/sd/stb_impl.cpp")
endif()

add_executable(chimera
    "${CMAKE_CURRENT_LIST_DIR}/chimera.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_whisper.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_sd.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/llama_build_info_shim.cpp"
    "${CHIMERA_STB_IMPL}"
)

target_include_directories(chimera PRIVATE
    "${CMAKE_SOURCE_DIR}/thirdparty"
    "${LLAMACPP_INCLUDE}"
    "${WHISPERCPP_INCLUDE}"
    "${SDCPP_INCLUDE}"
)

target_compile_options(chimera PRIVATE ${CXX_COMPILE_OPTIONS})
set_target_properties(chimera PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# Version strings — fall back to "unknown" if the parent did not set them.
foreach(_v CHIMERA_VERSION LLAMACPP_VERSION WHISPERCPP_VERSION SDCPP_VERSION)
    if(NOT DEFINED ${_v})
        set(${_v} "unknown")
    endif()
endforeach()
target_compile_definitions(chimera PRIVATE
    CHIMERA_VERSION="${CHIMERA_VERSION}"
    CHIMERA_LLAMACPP_VERSION="${LLAMACPP_VERSION}"
    CHIMERA_WHISPERCPP_VERSION="${WHISPERCPP_VERSION}"
    CHIMERA_SDCPP_VERSION="${SDCPP_VERSION}"
)

if(UNIX AND NOT APPLE)
    set(_CHIMERA_GGML_LIBS ${LIB_GGML} ${LIB_GGML_BASE} ${LIB_GGML_CPU})
    if(GGML_METAL)
        list(APPEND _CHIMERA_GGML_LIBS ${LIB_GGML_BLAS} ${LIB_GGML_METAL})
    endif()
    if(GGML_CUDA)
        list(APPEND _CHIMERA_GGML_LIBS ${LIB_GGML_CUDA})
    endif()
    if(GGML_VULKAN)
        list(APPEND _CHIMERA_GGML_LIBS ${LIB_GGML_VULKAN})
    endif()
    if(GGML_SYCL)
        list(APPEND _CHIMERA_GGML_LIBS ${LIB_GGML_SYCL})
    endif()
    if(GGML_HIP)
        list(APPEND _CHIMERA_GGML_LIBS ${LIB_GGML_HIP})
    endif()
    if(GGML_OPENCL)
        list(APPEND _CHIMERA_GGML_LIBS ${LIB_GGML_OPENCL})
    endif()

    target_link_libraries(chimera PRIVATE
        ${LIB_LLAMA}
        ${LIB_LLAMA_COMMON}
        ${LIB_MTMD}
        ${LIB_WHISPER_COMMON}
        ${LIB_WHISPER}
        ${LIB_SD}
        -Wl,--whole-archive
        ${_CHIMERA_GGML_LIBS}
        -Wl,--no-whole-archive
        ${SYSTEM_LIBS}
    )
elseif(APPLE)
    target_link_libraries(chimera PRIVATE
        ${LIB_LLAMA}
        ${LIB_LLAMA_COMMON}
        ${LIB_MTMD}
        ${LIB_WHISPER_COMMON}
        ${LIB_WHISPER}
        ${LIB_SD}
        ${SYSTEM_LIBS}
    )

    set(_CHIMERA_FORCE_LOAD_FLAGS
        "-Wl,-force_load,${LIB_GGML}"
        "-Wl,-force_load,${LIB_GGML_BASE}"
        "-Wl,-force_load,${LIB_GGML_CPU}"
    )
    if(GGML_METAL)
        list(APPEND _CHIMERA_FORCE_LOAD_FLAGS
            "-Wl,-force_load,${LIB_GGML_BLAS}"
            "-Wl,-force_load,${LIB_GGML_METAL}"
        )
    endif()
    if(GGML_CUDA)
        list(APPEND _CHIMERA_FORCE_LOAD_FLAGS "-Wl,-force_load,${LIB_GGML_CUDA}")
    endif()
    if(GGML_VULKAN)
        list(APPEND _CHIMERA_FORCE_LOAD_FLAGS "-Wl,-force_load,${LIB_GGML_VULKAN}")
    endif()
    if(GGML_SYCL)
        list(APPEND _CHIMERA_FORCE_LOAD_FLAGS "-Wl,-force_load,${LIB_GGML_SYCL}")
    endif()
    if(GGML_HIP)
        list(APPEND _CHIMERA_FORCE_LOAD_FLAGS "-Wl,-force_load,${LIB_GGML_HIP}")
    endif()
    if(GGML_OPENCL)
        list(APPEND _CHIMERA_FORCE_LOAD_FLAGS "-Wl,-force_load,${LIB_GGML_OPENCL}")
    endif()
    target_link_options(chimera PRIVATE ${_CHIMERA_FORCE_LOAD_FLAGS})
else()
    target_link_libraries(chimera PRIVATE
        ${LIB_LLAMA}
        ${LIB_LLAMA_COMMON}
        ${LIB_GGML}
        ${LIB_GGML_BASE}
        ${LIB_GGML_CPU}
        ${LIB_MTMD}
        ${LIB_WHISPER_COMMON}
        ${LIB_WHISPER}
        ${LIB_SD}
        ${SYSTEM_LIBS}
    )
endif()

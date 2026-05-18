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
    "${CMAKE_CURRENT_LIST_DIR}/chimera_chat_store.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_db.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_embed.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_embed_cache.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_pin_check.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_vector_store.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_serve.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_serve_rag.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_serve_chat_persist.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chimera_serve_chats_read.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/llama_build_info_shim.cpp"
    # server-http.cpp is the HTTP-frontend half of llama-server. Upstream
    # compiles it directly into the llama-server executable rather than into
    # libserver-context.a; we do the same, pulling the source out of the
    # llama.cpp tree via the src-aux/ directory that manage.py populates.
    "${LLAMACPP_SRC_AUX}/server-http.cpp"
    # SQLite + sqlite-vec amalgamations. Both are plain C, single TU, with
    # no external deps; we compile them straight into the chimera target.
    "${SQLITE_SRC_AUX}/sqlite3.c"
    "${SQLITE_VEC_SRC_AUX}/sqlite-vec.c"
)

# Optional modality sources. When the parent scope dropped a modality,
# its TUs are not compiled and the related routes/subcommands disappear
# (gated by CHIMERA_HAS_WHISPER / CHIMERA_HAS_SD compile defines that
# this block also sets). stb_image_write is only used by chimera_sd's
# PNG encoder, so the stb_impl TU follows SD on/off.
if(CHIMERA_LINK_WHISPER)
    target_sources(chimera PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/chimera_whisper.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/chimera_serve_audio.cpp"
    )
    target_compile_definitions(chimera PRIVATE CHIMERA_HAS_WHISPER)
endif()
if(CHIMERA_LINK_SD)
    target_sources(chimera PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/chimera_sd.cpp"
        "${CMAKE_CURRENT_LIST_DIR}/chimera_serve_images.cpp"
        "${CHIMERA_STB_IMPL}"
    )
    target_compile_definitions(chimera PRIVATE CHIMERA_HAS_SD)
endif()

target_include_directories(chimera PRIVATE
    "${CMAKE_SOURCE_DIR}/thirdparty"
    "${LLAMACPP_INCLUDE}"
    "${SQLITE_INCLUDE}"
    "${SQLITE_VEC_INCLUDE}"
)

# Forward the optional per-request payload cap (in bytes) to the locally-
# compiled server-http.cpp, which uses it to call srv->set_payload_max_length.
# Without this forwarding, the cache variable would be inert. Default 256 MB
# is encoded in server-http.cpp itself; this only fires when the user passes
# `-DCHIMERA_HTTP_PAYLOAD_MAX_BYTES=<bytes>` at configure time.
if(DEFINED CHIMERA_HTTP_PAYLOAD_MAX_BYTES)
    target_compile_definitions(chimera PRIVATE
        CHIMERA_HTTP_PAYLOAD_MAX_BYTES=${CHIMERA_HTTP_PAYLOAD_MAX_BYTES})
endif()
if(CHIMERA_LINK_WHISPER)
    target_include_directories(chimera PRIVATE "${WHISPERCPP_INCLUDE}")
endif()
if(CHIMERA_LINK_SD)
    target_include_directories(chimera PRIVATE "${SDCPP_INCLUDE}")
endif()

# SQLite compile-time tuning. The flags trim ~50 KB of dead code we don't
# use, force WAL-friendly defaults, and pre-enable FTS5 (needed by the chat
# search feature) + math functions (sqlite-vec uses some of these). See
# doc/dev/sqlite.md § 2 for the per-flag rationale.
#
# SQLITE_CORE switches sqlite-vec.h between extension-mode (where sqlite
# calls route through a sqlite3_api function-pointer table loaded at
# runtime) and core-mode (direct symbol calls). Since we statically link
# sqlite-vec.c into the same target as sqlite3.c, core-mode is correct.
target_compile_definitions(chimera PRIVATE
    SQLITE_CORE
    SQLITE_DQS=0
    SQLITE_DEFAULT_MEMSTATUS=0
    SQLITE_DEFAULT_WAL_SYNCHRONOUS=1
    SQLITE_LIKE_DOESNT_MATCH_BLOBS
    SQLITE_MAX_EXPR_DEPTH=0
    SQLITE_OMIT_DECLTYPE
    SQLITE_OMIT_DEPRECATED
    SQLITE_OMIT_PROGRESS_CALLBACK
    SQLITE_OMIT_SHARED_CACHE
    SQLITE_THREADSAFE=2
    SQLITE_ENABLE_FTS5
    SQLITE_ENABLE_MATH_FUNCTIONS
)

# Optional linenoise: header + macro flag are added only when the parent
# scope decided to link it. Linking happens in the platform blocks below.
if(CHIMERA_LINK_LINENOISE)
    target_include_directories(chimera PRIVATE "${LINENOISE_INCLUDE}")
    target_compile_definitions(chimera PRIVATE CHIMERA_HAS_LINENOISE)
endif()

# Optional embedded webui. When the top-level scope set CHIMERA_LINK_WEBUI,
# xxd-encode the four prebuilt assets (index.html, bundle.js, bundle.css,
# loading.html) into .hpp files under the chimera build dir, add them to
# the target sources, expose the generated dir on the include path so
# server-http.cpp can #include them, and flip the LLAMA_BUILD_WEBUI define
# that gates the upstream route-binding block in server-http.cpp.
if(CHIMERA_LINK_WEBUI)
    set(_webui_assets index.html bundle.js bundle.css loading.html)
    set(_webui_src "${LLAMACPP_SRC_AUX}/webui")
    set(_webui_xxd "${LLAMACPP_SRC_AUX}/xxd.cmake")
    foreach(_asset ${_webui_assets})
        set(_in  "${_webui_src}/${_asset}")
        set(_out "${CMAKE_CURRENT_BINARY_DIR}/${_asset}.hpp")
        add_custom_command(
            DEPENDS "${_in}" "${_webui_xxd}"
            OUTPUT  "${_out}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${_in}" "-DOUTPUT=${_out}" -P "${_webui_xxd}"
            COMMENT "xxd webui asset: ${_asset} -> ${_asset}.hpp"
            VERBATIM
        )
        set_source_files_properties("${_out}" PROPERTIES GENERATED TRUE)
        target_sources(chimera PRIVATE "${_out}")
    endforeach()
    target_include_directories(chimera PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(chimera PRIVATE LLAMA_BUILD_WEBUI)
    unset(_webui_assets)
    unset(_webui_src)
    unset(_webui_xxd)
endif()

# CXX_COMPILE_OPTIONS contains -std=c++17 (or /std:c++17 on MSVC), which
# the C compiler rejects. Gate the option list to C++ sources only so
# sqlite3.c / sqlite-vec.c / server-http.cpp (wait — that's C++) get
# only the flags they accept. server-http.cpp is .cpp so it's fine.
target_compile_options(chimera PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTIONS}>)
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

# Build-time intent baked in for `chimera info`. Each empty-string macro
# is filtered out at render time so we only print knobs that were
# actually set. See top-level CMakeLists for how these are derived.
target_compile_definitions(chimera PRIVATE
    CHIMERA_BUILT_BACKENDS="${CHIMERA_BUILT_BACKENDS}"
    CHIMERA_CUDA_ARCHITECTURES="${CHIMERA_CUDA_ARCHITECTURES}"
    CHIMERA_HIP_ARCHITECTURES="${CHIMERA_HIP_ARCHITECTURES}"
    CHIMERA_BLAS_VENDOR="${CHIMERA_BLAS_VENDOR}"
    CHIMERA_CUDA_FORCE_MMQ="${CHIMERA_CUDA_FORCE_MMQ}"
    CHIMERA_CUDA_FORCE_CUBLAS="${CHIMERA_CUDA_FORCE_CUBLAS}"
    CHIMERA_HIP_ROCWMMA_FATTN="${CHIMERA_HIP_ROCWMMA_FATTN}"
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

    # GNU ld is single-pass left-to-right: an archive only contributes
    # the members needed by *currently-unresolved* symbols. server-context
    # references symbols in llama-common (common_speculative_*,
    # common_remote_get_content, ...) that were already skipped over when
    # llama-common was first processed, because nothing earlier in the
    # link had asked for them. --start-group / --end-group makes ld
    # iterate the inner archives until they reach a fixed point, which
    # is exactly the case for archives with bidirectional references.
    # macOS's linker iterates by default so this is GNU-ld-only.
    set(_CHIMERA_MODALITY_LIBS "")
    if(CHIMERA_LINK_WHISPER)
        list(APPEND _CHIMERA_MODALITY_LIBS ${LIB_WHISPER_COMMON} ${LIB_WHISPER})
    endif()
    if(CHIMERA_LINK_SD)
        list(APPEND _CHIMERA_MODALITY_LIBS ${LIB_SD})
    endif()
    target_link_libraries(chimera PRIVATE
        -Wl,--start-group
        ${LIB_LLAMA}
        ${LIB_LLAMA_COMMON}
        ${LIB_MTMD}
        ${LIB_SERVER_CONTEXT}
        ${LIB_CPP_HTTPLIB}
        ${_CHIMERA_MODALITY_LIBS}
        -Wl,--end-group
        -Wl,--whole-archive
        ${_CHIMERA_GGML_LIBS}
        -Wl,--no-whole-archive
        ${SYSTEM_LIBS}
    )
    unset(_CHIMERA_MODALITY_LIBS)
elseif(APPLE)
    set(_CHIMERA_MODALITY_LIBS "")
    if(CHIMERA_LINK_WHISPER)
        list(APPEND _CHIMERA_MODALITY_LIBS ${LIB_WHISPER_COMMON} ${LIB_WHISPER})
    endif()
    if(CHIMERA_LINK_SD)
        list(APPEND _CHIMERA_MODALITY_LIBS ${LIB_SD})
    endif()
    target_link_libraries(chimera PRIVATE
        ${LIB_LLAMA}
        ${LIB_LLAMA_COMMON}
        ${LIB_MTMD}
        ${LIB_SERVER_CONTEXT}
        ${LIB_CPP_HTTPLIB}
        ${_CHIMERA_MODALITY_LIBS}
        ${SYSTEM_LIBS}
    )
    unset(_CHIMERA_MODALITY_LIBS)

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
    set(_CHIMERA_MODALITY_LIBS "")
    if(CHIMERA_LINK_WHISPER)
        list(APPEND _CHIMERA_MODALITY_LIBS ${LIB_WHISPER_COMMON} ${LIB_WHISPER})
    endif()
    if(CHIMERA_LINK_SD)
        list(APPEND _CHIMERA_MODALITY_LIBS ${LIB_SD})
    endif()
    target_link_libraries(chimera PRIVATE
        ${LIB_LLAMA}
        ${LIB_LLAMA_COMMON}
        ${LIB_GGML}
        ${LIB_GGML_BASE}
        ${LIB_GGML_CPU}
        ${LIB_MTMD}
        ${LIB_SERVER_CONTEXT}
        ${LIB_CPP_HTTPLIB}
        ${_CHIMERA_MODALITY_LIBS}
        ${SYSTEM_LIBS}
    )
    unset(_CHIMERA_MODALITY_LIBS)
endif()

if(CHIMERA_LINK_LINENOISE)
    target_link_libraries(chimera PRIVATE ${LIB_LINENOISE})
endif()

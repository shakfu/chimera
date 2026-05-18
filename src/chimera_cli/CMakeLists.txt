# -----------------------------------------------------------------------------
# chimera_cli - the chimera CLI executable. Just chimera.cpp linked against
# the chimera_lib STATIC target from src/chimera/. PUBLIC link properties
# on chimera_lib propagate the full transitive archive list and the
# platform whole-archive wrappers (-Wl,-force_load on macOS,
# --whole-archive on Linux, /WHOLEARCHIVE on Windows) that keep ggml
# backend static initializers alive, so this CMakeLists stays trivially
# small.
#
# Gated on CHIMERA_BUILD_EXECUTABLE (ON by default). A downstream consumer
# that only wants libchimera.a configures with
# -DCHIMERA_BUILD_EXECUTABLE=OFF to skip the CLI link entirely.
# -----------------------------------------------------------------------------

option(CHIMERA_BUILD_EXECUTABLE "Build the chimera CLI executable on top of libchimera.a" ON)

if(NOT CHIMERA_BUILD_EXECUTABLE)
    return()
endif()

if(SD_USE_VENDORED_GGML)
    # The library slice already returned out under this condition; mirror
    # the gate here so the executable doesn't try to link against a
    # chimera_lib target that doesn't exist.
    return()
endif()

add_executable(chimera
    "${CMAKE_CURRENT_LIST_DIR}/chimera.cpp"
    # GNU ld is single-pass: libchimera.a is linked ahead of the llama
    # archives, so when ld scans it, nothing yet references
    # llama_build_info() / llama_build_target() / llama_commit() /
    # llama_compiler() / llama_build_number(), and the shim TU inside
    # libchimera.a is skipped. Later, libllama-common.a (common.cpp) and
    # libserver-context.a (server-task.cpp) introduce those undefined
    # symbols, but the shim archive member is already past and not inside
    # the --start-group/--end-group block. Compiling the shim directly
    # into the executable puts it on the link line as a plain object,
    # which always contributes its symbols and satisfies those refs
    # regardless of archive order. macOS ld iterates archives by default
    # so it works there even without this, but the duplicate compile is
    # harmless (archive member is dropped when the symbols are already
    # defined).
    "${CMAKE_SOURCE_DIR}/src/chimera/llama_build_info_shim.cpp"
)
target_link_libraries(chimera PRIVATE chimera_lib)
target_compile_options(chimera PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_COMPILE_OPTIONS}>)
set_target_properties(chimera PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

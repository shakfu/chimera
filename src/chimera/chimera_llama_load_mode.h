#pragma once

// llama.cpp b10107 removed llama_model_params::use_mmap and ::use_mlock in
// favour of a single `enum llama_load_mode load_mode`. Chimera keeps the two
// booleans (they are what --no-mmap / --mlock set) and adds a `load_mode`
// string that names the enum directly, so the translation lives here. Kept as
// a leaf header — llama.h plus chimera.h, which has no llama dependency — so
// chimera_embed.cpp can share it without pulling in the rest of the llama glue
// in chimera_llama.h.

#include "chimera.h"  // ExitCode, fail
#include "llama.h"

#include <string>

// Names accepted by --load-mode, matching upstream's -lm/--load-mode values
// (common/arg.cpp). LLAMA_LOAD_MODE_DIRECT_IO is spelled "dio" there, and
// direct I/O silently falls back to a normal read where unsupported.
inline enum llama_load_mode chimera_parse_load_mode(const std::string & name) {
    if (name == "none")  return LLAMA_LOAD_MODE_NONE;
    if (name == "mmap")  return LLAMA_LOAD_MODE_MMAP;
    if (name == "mlock") return LLAMA_LOAD_MODE_MLOCK;
    if (name == "dio")   return LLAMA_LOAD_MODE_DIRECT_IO;
    fail(ExitCode::BadInput,
         "unknown --load-mode: " + name + " (use none|mmap|mlock|dio)");
}

// An explicit --load-mode wins outright; chimera cannot reproduce upstream's
// last-flag-wins ordering because the booleans and the enum live in separate
// option fields. When --load-mode is absent this mirrors upstream's
// deprecated-flag shim: --mlock selects LLAMA_LOAD_MODE_MLOCK (which is mmap +
// mlock) regardless of the mmap flag, otherwise mmap on/off picks between
// LLAMA_LOAD_MODE_MMAP and LLAMA_LOAD_MODE_NONE.
inline enum llama_load_mode chimera_llama_load_mode(const std::string & load_mode_name,
                                                    bool use_mmap,
                                                    bool use_mlock) {
    if (!load_mode_name.empty()) {
        return chimera_parse_load_mode(load_mode_name);
    }
    if (use_mlock) {
        return LLAMA_LOAD_MODE_MLOCK;
    }
    return use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
}

// chimera_serve_meta.cpp — chimera-specific meta endpoints.
//
// Three small handlers that don't belong with any of the modality
// modules (audio, images, RAG, chat persistence):
//
//   GET  /v1/chimera/info   — JSON snapshot of what `chimera info` prints
//                             (versions, built/loaded backends, devices,
//                             CPU features, build flags). For an About
//                             pane in a desktop client (chimera-desktop).
//   GET  /v1/chimera/db     — JSON snapshot of what `chimera db status`
//                             prints (path, file size, schema version,
//                             tables + per-table row counts). For a
//                             "DB used: X MB" footer in a chat browser.
//   POST /v1/chimera/shutdown — graceful exit. Replies 202 then triggers
//                             the same teardown the SIGINT handler does.
//                             Used by chimera-desktop's parent process
//                             to clean up the child before SIGKILL.
//
// Each is small and self-contained. We deliberately re-derive the data
// (versions / device info / sqlite stats) inline rather than refactoring
// chimera.cpp's `command_info` / `command_db_status` into a shared helper —
// those CLI commands stream to std::cout while the JSON form here builds
// a structured tree, and the duplication is bounded (~150 LOC total).
// If the JSON shape ever needs to mirror the CLI text exactly, factor
// the data-gathering out then.

#include "chimera_serve_internal.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#ifdef CHIMERA_HAS_WHISPER
#  include "chimera_whisper.h"
#endif
#ifdef CHIMERA_HAS_SD
#  include "chimera_sd.h"
#endif

#include <chrono>
#include <filesystem>
#include <memory>
#include <sqlite3.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace chimera_serve {

namespace {

server_http_res_ptr meta_err(int code, const std::string & msg) {
    auto r = std::make_unique<server_http_res>();
    r->status = code;
    r->data = json{{ "error",
        { { "message", msg }, { "code", code }, { "type", "invalid_request_error" } }
    }}.dump();
    return r;
}

// --- ggml backend / device descriptors ---------------------------------

const char * dev_type_label(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        default:                             return "?";
    }
}

// Primary backend label: the first non-CPU registry name registered, or
// "CPU" if no accelerator registry is present. Mirrors the heuristic in
// chimera.cpp's `primary_backend_label()`.
std::string primary_backend_label() {
    const size_t n_reg = ggml_backend_reg_count();
    for (size_t i = 0; i < n_reg; ++i) {
        const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i));
        if (name && std::string(name) != "CPU") return name;
    }
    return "CPU";
}

json backend_registries_json() {
    json out = json::array();
    const size_t n_reg = ggml_backend_reg_count();
    for (size_t i = 0; i < n_reg; ++i) {
        const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i));
        out.push_back(name ? name : "?");
    }
    return out;
}

json devices_json() {
    json out = json::array();
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        auto * d = ggml_backend_dev_get(i);
        const char * dn = ggml_backend_dev_name(d);
        const char * dd = ggml_backend_dev_description(d);
        out.push_back({
            { "name",        dn ? dn : "?" },
            { "type",        dev_type_label(ggml_backend_dev_type(d)) },
            { "description", dd ? dd : "" },
        });
    }
    return out;
}

// "macOS-arm64" / "Linux-x86_64" / "Windows-amd64" etc. Mirrors
// chimera.cpp's `platform_label()` without the CLI dependency.
std::string platform_label_str() {
    std::string os =
#if defined(__APPLE__)
        "macOS"
#elif defined(__linux__)
        "Linux"
#elif defined(_WIN32)
        "Windows"
#else
        "unknown"
#endif
        ;
    std::string arch =
#if defined(__aarch64__) || defined(_M_ARM64)
        "arm64"
#elif defined(__x86_64__) || defined(_M_X64)
        "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
        "x86"
#else
        "unknown"
#endif
        ;
    return os + "-" + arch;
}

// Parse "[NAME ON ... | name on | ... ] | feat=1 | feat2=1" into
// just the feature names. Matches chimera.cpp::parse_sys_info but
// avoids depending on it.
std::vector<std::string> parse_cpu_features(const std::string & info) {
    std::vector<std::string> out;
    size_t pos = 0;
    while (pos < info.size()) {
        size_t bar = info.find('|', pos);
        std::string token =
            (bar == std::string::npos) ? info.substr(pos) : info.substr(pos, bar - pos);
        // trim
        size_t a = token.find_first_not_of(" \t");
        size_t b = token.find_last_not_of(" \t");
        if (a != std::string::npos) {
            token = token.substr(a, b - a + 1);
            // accept "FEAT = 1" / "FEAT=1" / "FEAT = on" forms
            size_t eq = token.find('=');
            if (eq != std::string::npos) {
                std::string key = token.substr(0, eq);
                std::string val = token.substr(eq + 1);
                size_t ka = key.find_first_not_of(" \t");
                size_t kb = key.find_last_not_of(" \t");
                size_t va = val.find_first_not_of(" \t");
                if (ka != std::string::npos && va != std::string::npos) {
                    std::string k = key.substr(ka, kb - ka + 1);
                    char first = val[va];
                    if (first == '1' || first == 'o' || first == 'O' || first == 't' || first == 'T') {
                        out.push_back(k);
                    }
                }
            }
        }
        if (bar == std::string::npos) break;
        pos = bar + 1;
    }
    return out;
}

// Split comma-separated CHIMERA_BUILT_BACKENDS into a JSON array.
json built_backends_json() {
    json out = json::array();
    std::string s = CHIMERA_BUILT_BACKENDS;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t c = s.find(',', pos);
        std::string tok = (c == std::string::npos) ? s.substr(pos) : s.substr(pos, c - pos);
        if (!tok.empty()) out.push_back(tok);
        if (c == std::string::npos) break;
        pos = c + 1;
    }
    return out;
}

}  // namespace

// ----------------------------------------------------------------------------
// GET /v1/chimera/info
// ----------------------------------------------------------------------------

server_http_context::handler_t make_chimera_info_handler() {
    return [](const server_http_req &) -> server_http_res_ptr {
        json info;
        info["object"]   = "chimera.info";
        info["chimera"]  = {
            { "version",  CHIMERA_VERSION },
            { "platform", platform_label_str() },
        };
        info["llama_cpp"] = {
            { "version",        CHIMERA_LLAMACPP_VERSION },
            { "ggml_version",   ggml_version() },
            { "ggml_commit",    ggml_commit() },
            { "built_backends", built_backends_json() },
            { "loaded_backend", primary_backend_label() },
            { "registries",     backend_registries_json() },
            { "devices",        devices_json() },
            { "gpu_offload",    llama_supports_gpu_offload() },
            { "mmap_support",   llama_supports_mmap() },
            { "mlock_support",  llama_supports_mlock() },
            { "rpc_support",    llama_supports_rpc() },
        };

#ifdef CHIMERA_HAS_WHISPER
        info["whisper_cpp"] = {
            { "linked",       true },
            { "version",      CHIMERA_WHISPERCPP_VERSION },
            { "ggml_version", chimera_whisper::whisper_ggml_version() },
            { "cpu_features", parse_cpu_features(chimera_whisper::whisper_system_info_raw()) },
        };
#else
        info["whisper_cpp"] = { { "linked", false } };
#endif

#ifdef CHIMERA_HAS_SD
        info["stable_diffusion_cpp"] = {
            { "linked",       true },
            { "version",      CHIMERA_SDCPP_VERSION },
            { "ggml_version", chimera_sd::sd_ggml_version() },
            { "cpu_features", parse_cpu_features(chimera_sd::sd_system_info_raw()) },
        };
#else
        info["stable_diffusion_cpp"] = { { "linked", false } };
#endif

        info["sqlite"] = {
            { "version",    chimera_db::sqlite_version() },
            { "sqlite_vec", chimera_db::sqlite_vec_version() },
        };

        json flags = json::object();
        auto add_flag = [&](const char * key, const char * value) {
            if (value && *value) flags[key] = value;
        };
        add_flag("CUDA_ARCH",         CHIMERA_CUDA_ARCHITECTURES);
        add_flag("HIP_ARCH",          CHIMERA_HIP_ARCHITECTURES);
        add_flag("BLAS_VENDOR",       CHIMERA_BLAS_VENDOR);
        add_flag("CUDA_FORCE_MMQ",    CHIMERA_CUDA_FORCE_MMQ);
        add_flag("CUDA_FORCE_CUBLAS", CHIMERA_CUDA_FORCE_CUBLAS);
        add_flag("HIP_ROCWMMA_FATTN", CHIMERA_HIP_ROCWMMA_FATTN);
        info["build_flags"] = flags;

        auto res = std::make_unique<server_http_res>();
        res->data = info.dump();
        return res;
    };
}

// ----------------------------------------------------------------------------
// GET /v1/chimera/db
// ----------------------------------------------------------------------------

server_http_context::handler_t make_chimera_db_handler(const std::string & db_path_override) {
    const std::string path_override = db_path_override;
    return [path_override](const server_http_req &) -> server_http_res_ptr {
        const std::string path = path_override.empty()
            ? chimera_db::default_path()
            : path_override;

        chimera_db::Connection conn;
        try {
            conn = chimera_db::open_and_migrate(path);
        } catch (const std::exception & e) {
            return meta_err(500, std::string("failed to open db: ") + e.what());
        }

        // File size from the filesystem. Returns 0 if the DB doesn't
        // exist on disk yet (in-memory or freshly migrated and empty
        // — open_and_migrate would have written something, but be
        // defensive).
        int64_t size_bytes = 0;
        try {
            if (std::filesystem::exists(path)) {
                size_bytes = static_cast<int64_t>(std::filesystem::file_size(path));
            }
        } catch (const std::exception &) {
            size_bytes = 0;
        }

        const int sv  = chimera_db::current_schema_version(conn.get());
        const int sv_target = chimera_db::latest_schema_version();
        const auto tables = chimera_db::list_tables(conn.get());

        // Per-table row counts. Cheap (COUNT(*) on each); the DB is
        // expected to be small for chimera's use case.
        json row_counts = json::object();
        for (const auto & t : tables) {
            // Validate table name — only [A-Za-z0-9_] — before
            // splicing into SQL. list_tables() returns names from
            // sqlite_master, so they're already safe, but defense in
            // depth.
            bool safe = !t.empty();
            for (char c : t) {
                if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                      (c >= '0' && c <= '9') || c == '_')) { safe = false; break; }
            }
            if (!safe) continue;
            const std::string sql = "SELECT COUNT(*) FROM \"" + t + "\"";
            sqlite3_stmt * stmt = nullptr;
            if (sqlite3_prepare_v2(conn.get(), sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
                if (sqlite3_step(stmt) == SQLITE_ROW) {
                    row_counts[t] = sqlite3_column_int64(stmt, 0);
                }
                sqlite3_finalize(stmt);
            }
        }

        json out = {
            { "object",                    "chimera.db" },
            { "path",                      path },
            { "size_bytes",                size_bytes },
            { "sqlite_version",            chimera_db::sqlite_version() },
            { "sqlite_vec_version",        chimera_db::sqlite_vec_version() },
            { "sqlite_vec_loaded_version", chimera_db::sqlite_vec_loaded_version(conn.get()) },
            { "schema_version",            sv },
            { "latest_schema_version",     sv_target },
            { "tables",                    tables },
            { "row_counts",                row_counts },
        };

        auto res = std::make_unique<server_http_res>();
        res->data = out.dump();
        return res;
    };
}

// ----------------------------------------------------------------------------
// POST /v1/chimera/shutdown
// ----------------------------------------------------------------------------
//
// The `trigger` lambda is whatever `command_serve` would have run on
// SIGINT (same as `g_shutdown_handler`). The handler queues the response
// first and then runs `trigger` on a detached thread after a short
// delay so the 202 actually reaches the client before the listener
// stops accepting connections.

server_http_context::handler_t make_chimera_shutdown_handler(std::function<void()> trigger) {
    return [trigger](const server_http_req &) -> server_http_res_ptr {
        if (trigger) {
            std::thread([trigger] {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                trigger();
            }).detach();
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 202;
        res->data = json{
            { "object",  "chimera.shutdown" },
            { "status",  "shutting_down" },
            { "delay_ms", 150 }
        }.dump();
        return res;
    };
}

}  // namespace chimera_serve

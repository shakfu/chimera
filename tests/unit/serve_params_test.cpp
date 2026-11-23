// serve_params_test.cpp - pin the common_params that `chimera serve` hands
// upstream for a default ServeOptions. chimera bypasses upstream's arg parser,
// so an upstream default change reaches chimera silently; this turns it into
// a failure. A failing line is a decision: set the field in
// build_common_params(), or accept the new default here.

#include "chimera_serve_internal.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    const ServeOptions   opts;
    const common_params  p = chimera_serve::build_common_params(opts);

    // Security: who can reach the server and what a request can read.
    check(p.cors_origins == "localhost", "cors_origins == \"localhost\" (CSRF + rebinding checks on)");
    check(p.cors_origins_explicit, "cors_origins_explicit");
    check(p.cors_credentials, "cors_credentials == true (upstream default)");
    check(p.hostnames.size() == 1 && p.hostnames[0] == "127.0.0.1", "hostnames == {127.0.0.1}");
    check(p.api_keys.empty(), "api_keys empty without --api-key");
    check(p.api_prefix.empty(), "api_prefix empty (chimera routes assume none)");
    check(p.media_path.empty(), "media_path empty (file:// media refused)");
    check(!p.media_allow_private_urls, "media_allow_private_urls == false (SSRF guard)");
    check(!p.endpoint_props, "endpoint_props == false (POST /props refused)");
    check(p.ssl_file_key.empty() && p.ssl_file_cert.empty(), "no TLS files by default");
    check(p.models_dir.empty() && p.models_preset.empty(), "router mode off");

    // Behavior chimera relies on without setting.
    check(p.endpoint_slots, "endpoint_slots == true (GET /slots)");
    check(p.endpoint_metrics, "endpoint_metrics == true");
    check(p.n_threads_http < 1, "n_threads_http < 1 (server-http.cpp sizes the pool)");
    check(p.timeout_read == 3600 && p.timeout_write == 3600, "HTTP timeouts 3600 s");
    check(p.sse_ping_interval == 30, "sse_ping_interval == 30 s");
    check(p.sleep_idle_seconds <= 0, "sleep_idle_seconds off");
    check(p.slot_save_path.empty(), "slot_save_path empty (POST /slots/:id refused)");

    // Values build_common_params() must resolve itself.
    check(p.cpuparams.n_threads > 0, "cpuparams.n_threads resolved from -1");
    check(p.cpuparams_batch.n_threads > 0, "cpuparams_batch.n_threads resolved");
    check(!p.tensor_buft_overrides.empty() && p.tensor_buft_overrides.back().pattern == nullptr,
          "tensor_buft_overrides null-terminated");

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("serve_params_test: all checks passed\n");
    return 0;
}

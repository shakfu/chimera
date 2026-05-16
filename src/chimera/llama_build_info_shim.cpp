int llama_build_number(void) {
    return 0;
}

const char * llama_commit(void) {
    return "unknown";
}

const char * llama_compiler(void) {
    return "unknown";
}

const char * llama_build_target(void) {
    return "chimera";
}

// Referenced by libserver-context.a (server-task.cpp's
// to_json_oaicompat_chat_stream) and libllama-common.a (download.cpp,
// hf-cache.cpp). Upstream generates this string at configure time via
// build-info.cpp.in; here we return a stable identifier so /v1/models
// and User-Agent headers carry a recognizable value instead of garbage.
const char * llama_build_info(void) {
    return "chimera";
}

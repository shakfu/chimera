// chimera_serve_audio.cpp — POST /v1/audio/{transcriptions,translations} handlers
// (whisper.cpp behind the /v1/audio surface). Bound from chimera_serve.cpp
// when --enable-audio loaded a model. Extracted from chimera_serve.cpp.

#include "chimera_serve_internal.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace chimera_serve {

namespace {

std::string srt_timestamp(int64_t t_10ms) {
    int64_t ms = t_10ms * 10;
    int64_t h = ms / 3600000; ms %= 3600000;
    int64_t m = ms / 60000;   ms %= 60000;
    int64_t s = ms / 1000;    ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld,%03lld",
                  (long long)h, (long long)m, (long long)s, (long long)ms);
    return buf;
}

std::string vtt_timestamp(int64_t t_10ms) {
    int64_t ms = t_10ms * 10;
    int64_t h = ms / 3600000; ms %= 3600000;
    int64_t m = ms / 60000;   ms %= 60000;
    int64_t s = ms / 1000;    ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld.%03lld",
                  (long long)h, (long long)m, (long long)s, (long long)ms);
    return buf;
}

std::string format_srt(const chimera_whisper::TranscribeResult & r) {
    std::ostringstream out;
    for (size_t i = 0; i < r.segments.size(); ++i) {
        out << (i + 1) << '\n'
            << srt_timestamp(r.segments[i].t0) << " --> "
            << srt_timestamp(r.segments[i].t1) << '\n'
            << r.segments[i].text << "\n\n";
    }
    return out.str();
}

std::string format_vtt(const chimera_whisper::TranscribeResult & r) {
    std::ostringstream out;
    out << "WEBVTT\n\n";
    for (const auto & s : r.segments) {
        out << vtt_timestamp(s.t0) << " --> "
            << vtt_timestamp(s.t1) << '\n'
            << s.text << "\n\n";
    }
    return out.str();
}

json format_verbose_json(const chimera_whisper::TranscribeResult & r,
                         const std::string & task,
                         const std::string & language,
                         bool                include_words) {
    json segs = json::array();
    // OpenAI's "verbose_json" emits per-word entries at the top level (one
    // flat array of {word, start, end}), not inside each segment. We do
    // the same — agrees with what python-openai expects when
    // `timestamp_granularities=["word"]` is passed.
    json all_words = json::array();

    for (size_t i = 0; i < r.segments.size(); ++i) {
        // OpenAI's spec uses fractional-seconds (double) for start/end.
        // 10ms units -> seconds via /100.0.
        segs.push_back({
            { "id",    static_cast<int>(i) },
            { "seek",  0 },
            { "start", r.segments[i].t0 / 100.0 },
            { "end",   r.segments[i].t1 / 100.0 },
            { "text",  r.segments[i].text },
        });

        if (include_words) {
            for (const auto & w : r.segments[i].words) {
                all_words.push_back({
                    { "word",  w.text },
                    { "start", w.t0 / 100.0 },
                    { "end",   w.t1 / 100.0 },
                });
            }
        }
    }
    json out = {
        { "task",     task },
        { "language", language },
        { "duration", r.audio_duration_s },
        { "text",     r.text },
        { "segments", segs },
    };
    if (include_words) {
        out["words"] = std::move(all_words);
    }
    return out;
}

}  // namespace

// Build the handler bound to /v1/audio/{transcriptions,translations}.
// Captures the per-server whisper context + mutex by reference; both
// live in command_serve for as long as the handler can be invoked, so
// the reference is safe.
//
// `translate=true` is the difference between the two routes: whisper
// emits the source language verbatim for transcriptions, or English
// regardless of the input language for translations (whisper's built-in
// translate mode). Everything else is shared, so the two routes bind
// the same handler factory with different translate flags.
//
// SUPPORTED audio formats: WAV (RIFF/WAVE, PCM 8/16/24/32-bit integer,
// or 32-bit float). The OpenAI spec also accepts mp3, mp4, mpeg, mpga,
// m4a, webm — those need a real decoder (libsndfile + libavcodec or a
// single-header dr_mp3/dr_flac); not yet wired.
//
// SUPPORTED request fields: file (required), model (ignored, we have one),
// language, prompt (-> initial_prompt), response_format (json/text/
// verbose_json/srt/vtt), temperature (currently ignored — whisper doesn't
// expose a temperature knob in the same sense). timestamp_granularities is
// also ignored; segment-level timing is always returned in verbose_json.
server_http_context::handler_t make_audio_transcribe_handler(
    whisper_context  * ctx,
    std::mutex       & ctx_mutex,
    bool               translate,
    const std::string & vad_model_path) {

    return [ctx, &ctx_mutex, translate, vad_model_path](const server_http_req & req) -> server_http_res_ptr {
        auto err_res = [](int code, const std::string & msg) {
            auto res = std::make_unique<server_http_res>();
            res->status = code;
            res->data = json{{ "error", { { "message", msg }, { "code", code }, { "type", "invalid_request_error" }}}}.dump();
            return res;
        };

        // Audio bytes: the multipart "file" field, as deposited by
        // server-http.cpp's MultipartFormDataMap reader.
        auto file_it = req.files.find("file");
        if (file_it == req.files.end() || file_it->second.data.empty()) {
            return err_res(400, "missing 'file' field in multipart form");
        }
        const auto & upload = file_it->second;

        // Text fields: server-http folds them into a JSON object stored in
        // req.body. Tolerate the empty / non-JSON body case (e.g. file-only
        // request) by treating it as an empty object.
        json fields = json::object();
        if (!req.body.empty()) {
            try {
                fields = json::parse(req.body);
            } catch (const std::exception &) {
                fields = json::object();
            }
        }

        const std::string fmt = fields.value("response_format", std::string("json"));
        const std::string lang = fields.value("language", std::string("auto"));
        const std::string prompt = fields.value("prompt", std::string());

        // OpenAI's `timestamp_granularities` is an array of strings;
        // `["word"]` (or `["segment", "word"]`) turns on per-word
        // timing. The default is segment-only, which is what we already
        // emit. Tolerate the field being a string instead of array (some
        // clients pass it that way) or being absent entirely.
        // The OpenAI Python SDK + curl `-F 'timestamp_granularities[]=word'`
        // send the form key with PHP-style `[]` brackets; JSON bodies use
        // the bare key. Accept either, and tolerate string-or-array shape.
        bool want_word_ts = false;
        auto check_granularity = [&](const json & g) {
            if (g.is_array()) {
                for (const auto & v : g) {
                    if (v.is_string() && v.get<std::string>() == "word") {
                        want_word_ts = true;
                        return;
                    }
                }
            } else if (g.is_string() && g.get<std::string>() == "word") {
                want_word_ts = true;
            }
        };
        if (fields.contains("timestamp_granularities")) {
            check_granularity(fields["timestamp_granularities"]);
        }
        if (!want_word_ts && fields.contains("timestamp_granularities[]")) {
            check_granularity(fields["timestamp_granularities[]"]);
        }

        // Decode WAV. Only RIFF/WAVE is supported; mp3, m4a, webm, etc.
        // need a follow-up decoder. The error message names the
        // limitation so callers don't have to read the source.
        chimera_whisper::WavData wav;
        try {
            wav = chimera_whisper::load_wav_bytes(upload.data.data(), upload.data.size());
        } catch (const ChimeraError & e) {
            return err_res(415,
                std::string("unsupported audio: ") + e.what() +
                " (chimera accepts WAV / RIFF only by design; transcode mp3/m4a/webm to WAV with `ffmpeg -i in.<ext> -ar 16000 -ac 1 out.wav` before uploading)");
        }
        auto audio = chimera_whisper::resample_linear(
            wav.samples, wav.sample_rate, /*WHISPER_SAMPLE_RATE=*/16000);

        // Audio wave 1 — read the per-request tuning knobs that already
        // exist on TranscribeRequest. Each field is read only when
        // present so the sentinel defaults on TranscribeRequest take
        // effect for omitted keys. The coerce_* family (incl. bool)
        // lives in chimera_serve_internal.h; the image handler shares it.
        chimera_whisper::TranscribeRequest treq;
        treq.audio_16k_mono  = std::move(audio);
        treq.language        = lang;
        treq.translate       = translate;
        treq.no_context      = false;
        treq.emit_timestamps = true;   // needed for verbose_json/srt/vtt
        treq.initial_prompt  = prompt;
        treq.word_timestamps = want_word_ts;
        // Decoding strategy (per-request).
        if (fields.contains("temperature"))
            treq.temperature = coerce_float(fields["temperature"], treq.temperature);
        if (fields.contains("beam_size"))
            treq.beam_size   = coerce_int  (fields["beam_size"],   treq.beam_size);
        if (fields.contains("best_of"))
            treq.best_of     = coerce_int  (fields["best_of"],     treq.best_of);
        if (fields.contains("no_fallback"))
            treq.no_fallback = coerce_bool (fields["no_fallback"], treq.no_fallback);
        // Region of audio (slice the upload without re-encoding).
        if (fields.contains("offset_ms"))
            treq.offset_ms   = coerce_int  (fields["offset_ms"],   treq.offset_ms);
        if (fields.contains("duration_ms"))
            treq.duration_ms = coerce_int  (fields["duration_ms"], treq.duration_ms);
        // Segment shaping (pairs naturally with response_format=srt/vtt).
        if (fields.contains("max_len"))
            treq.max_len       = coerce_int (fields["max_len"],       treq.max_len);
        if (fields.contains("split_on_word"))
            treq.split_on_word = coerce_bool(fields["split_on_word"], treq.split_on_word);
        // Audio wave 2 — decoder-fail thresholds. NaN sentinels on
        // TranscribeRequest are preserved when the field is omitted (no
        // negative-as-sentinel here because `logprob_thold`'s upstream
        // default is itself negative — only NaN unambiguously means
        // "leave whisper's default"). `no_fallback=true` still wins
        // and clamps `temperature_inc` negative at the engine layer.
        if (fields.contains("temperature_inc"))
            treq.temperature_inc = coerce_float(fields["temperature_inc"], treq.temperature_inc);
        if (fields.contains("entropy_thold"))
            treq.entropy_thold   = coerce_float(fields["entropy_thold"],   treq.entropy_thold);
        if (fields.contains("logprob_thold"))
            treq.logprob_thold   = coerce_float(fields["logprob_thold"],   treq.logprob_thold);
        if (fields.contains("no_speech_thold"))
            treq.no_speech_thold = coerce_float(fields["no_speech_thold"], treq.no_speech_thold);

        // VAD bundle (step 5a). vad=true activates whisper's VAD
        // preprocessor; it needs a VAD model path on the request.
        // The model path is server-init (--audio-vad-model) — we don't
        // accept it from the client to avoid the server reading
        // arbitrary filesystem paths from external callers. A request
        // with vad=true but no server-configured VAD model returns 400
        // before transcription runs.
        if (fields.contains("vad") && coerce_bool(fields["vad"], false)) {
            if (vad_model_path.empty()) {
                return err_res(400,
                    "vad=true requires the server to have been started with "
                    "--audio-vad-model <path>; this server has no VAD model loaded");
            }
            treq.vad            = true;
            treq.vad_model_path = vad_model_path;
            if (fields.contains("vad_threshold"))
                treq.vad_threshold = coerce_float(fields["vad_threshold"], treq.vad_threshold);
            if (fields.contains("vad_min_speech_duration_ms"))
                treq.vad_min_speech_duration_ms = coerce_int(fields["vad_min_speech_duration_ms"], treq.vad_min_speech_duration_ms);
            if (fields.contains("vad_min_silence_duration_ms"))
                treq.vad_min_silence_duration_ms = coerce_int(fields["vad_min_silence_duration_ms"], treq.vad_min_silence_duration_ms);
            if (fields.contains("vad_max_speech_duration_s"))
                treq.vad_max_speech_duration_s = coerce_float(fields["vad_max_speech_duration_s"], treq.vad_max_speech_duration_s);
            if (fields.contains("vad_speech_pad_ms"))
                treq.vad_speech_pad_ms = coerce_int(fields["vad_speech_pad_ms"], treq.vad_speech_pad_ms);
            if (fields.contains("vad_samples_overlap"))
                treq.vad_samples_overlap = coerce_float(fields["vad_samples_overlap"], treq.vad_samples_overlap);
        }

        // Stereo speaker diarization. The CLI does this in command_whisper
        // — same algorithm, ported here. Each channel is resampled to
        // 16 kHz independently; per-segment energy ratio (1.1× threshold)
        // picks the label. Mono input + diarize=true is a 400 BadRequest.
        const bool want_diarize =
            fields.contains("diarize") && coerce_bool(fields["diarize"], false);
        std::vector<std::vector<float>> diarize_16k;
        if (want_diarize) {
            if (wav.channels != 2 || wav.per_channel.size() != 2) {
                return err_res(400,
                    "diarize=true requires a 2-channel (stereo) WAV upload; "
                    "got " + std::to_string(wav.channels) + "-channel audio");
            }
            diarize_16k.resize(2);
            for (int c = 0; c < 2; ++c) {
                diarize_16k[c] = chimera_whisper::resample_linear(
                    wav.per_channel[c], wav.sample_rate, /*WHISPER_SAMPLE_RATE=*/16000);
            }
        }
        auto estimate_speaker = [&](int64_t t0, int64_t t1) -> std::string {
            if (diarize_16k.size() != 2) return "";
            const int64_t n_samples = static_cast<int64_t>(diarize_16k[0].size());
            const auto clamp = [&](int64_t t) -> int64_t {
                int64_t s = (t * 16000) / 100;   // 10ms units → 16 kHz samples
                if (s < 0)         s = 0;
                if (s > n_samples) s = n_samples;
                return s;
            };
            const int64_t is0 = clamp(t0);
            const int64_t is1 = clamp(t1);
            double e0 = 0.0, e1 = 0.0;
            for (int64_t j = is0; j < is1; ++j) {
                e0 += std::fabs(diarize_16k[0][j]);
                e1 += std::fabs(diarize_16k[1][j]);
            }
            const char * id = (e0 > 1.1 * e1) ? "0" : (e1 > 1.1 * e0) ? "1" : "?";
            return std::string("(speaker ") + id + ")";
        };

        chimera_whisper::TranscribeResult result;
        {
            // whisper_full mutates internal state on the shared context, so
            // serialize concurrent transcription requests. Slow path anyway.
            std::lock_guard<std::mutex> lock(ctx_mutex);
            try {
                result = chimera_whisper::transcribe(ctx, treq);
            } catch (const ChimeraError & e) {
                return err_res(500, std::string("transcription failed: ") + e.what());
            }
        }

        // Stamp segments with speaker labels for the chosen response_format.
        // Mirrors command_whisper: write to Segment.speaker (structured)
        // and prefix Segment.text so the SRT/VTT/verbose_json/text writers
        // render the label without needing format-specific changes.
        if (want_diarize) {
            for (auto & s : result.segments) {
                s.speaker = estimate_speaker(s.t0, s.t1);
                if (!s.speaker.empty()) s.text = s.speaker + ' ' + s.text;
            }
            // result.text is the joined transcript; rebuild it so the
            // plain-text response_format also reflects the speakers.
            std::string joined;
            for (const auto & s : result.segments) {
                if (!joined.empty()) joined += ' ';
                joined += s.text;
            }
            result.text = std::move(joined);
        }

        // Response language: detected language if "auto" was requested,
        // otherwise echo the user's choice.
        const std::string out_lang =
            result.detected_language.empty() ? lang : result.detected_language;

        auto res = std::make_unique<server_http_res>();
        res->status = 200;

        if (fmt == "text") {
            res->content_type = "text/plain; charset=utf-8";
            res->data = result.text;
        } else if (fmt == "srt") {
            res->content_type = "application/x-subrip; charset=utf-8";
            res->data = format_srt(result);
        } else if (fmt == "vtt") {
            res->content_type = "text/vtt; charset=utf-8";
            res->data = format_vtt(result);
        } else if (fmt == "verbose_json") {
            res->data = format_verbose_json(
                result, translate ? "translate" : "transcribe", out_lang,
                want_word_ts).dump();
        } else {
            // Default: simple {"text": "..."} per OpenAI spec.
            res->data = json{{ "text", result.text }}.dump();
        }
        return res;
    };
}

}  // namespace chimera_serve

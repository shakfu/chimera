#!/usr/bin/env bash
#
# Smoke + end-to-end tests for chimera.
#
# Smoke tests always run (no model files needed): exercise --version,
# --help, and per-subcommand --help. They catch link-time regressions and
# CLI11 wiring breakage.
#
# End-to-end tests run only when the matching model file is present under
# models/. Missing models are reported as SKIP, not FAIL, so the test
# target stays green on a fresh checkout.
#
# Usage:
#   scripts/test.sh           # smoke + e2e (where models available)
#   scripts/test.sh --smoke   # smoke only
#
# Set CHIMERA=/path/to/chimera to test a non-default binary.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$REPO_ROOT/models"
SMOKE_ONLY=0
[[ "${1:-}" == "--smoke" ]] && SMOKE_ONLY=1

# Colorized result tags. Disabled when stdout isn't a tty (so logs
# captured via `make test 2>&1 | tee` stay plain text) and when the
# user sets NO_COLOR=1 (de-facto standard, see no-color.org). ANSI
# escapes are folded directly into the format string of each printf
# below; they contain no `%` so they don't interact with arg counts.
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    _GREEN=$'\033[32m'; _RED=$'\033[31m'; _YELLOW=$'\033[33m'; _BOLD=$'\033[1m'; _RESET=$'\033[0m'
else
    _GREEN=""; _RED=""; _YELLOW=""; _BOLD=""; _RESET=""
fi
PASS_TAG="${_GREEN}${_BOLD}PASS${_RESET}"
FAIL_TAG="${_RED}${_BOLD}FAIL${_RESET}"
SKIP_TAG="${_YELLOW}${_BOLD}SKIP${_RESET}"

# Locate the built binary. Single-config generators (Unix Makefiles,
# Ninja) put it directly at build/chimera; multi-config generators
# (MSBuild, Xcode) put it under build/<Config>/chimera[.exe].
locate_chimera() {
    local candidates=(
        "$REPO_ROOT/build/chimera"
        "$REPO_ROOT/build/chimera.exe"
        "$REPO_ROOT/build/Release/chimera.exe"
        "$REPO_ROOT/build/Release/chimera"
        "$REPO_ROOT/build/Debug/chimera.exe"
        "$REPO_ROOT/build/Debug/chimera"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "$c" || -f "$c" ]]; then
            printf '%s' "$c"
            return 0
        fi
    done
    return 1
}

CHIMERA="${CHIMERA:-$(locate_chimera || true)}"

if [[ -z "$CHIMERA" || ! -f "$CHIMERA" ]]; then
    printf "${FAIL_TAG}: chimera binary not found under %s/build/\n" "$REPO_ROOT" >&2
    echo  "       searched: build/chimera, build/chimera.exe, build/{Release,Debug}/chimera[.exe]" >&2
    echo  "       (run 'make build' first, or set CHIMERA=...)" >&2
    exit 1
fi

pass=0
fail=0
skip=0
failed_names=()

run() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf "  ${PASS_TAG}  %s\n" "$name"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  %s\n" "$name"
        fail=$((fail + 1))
        failed_names+=("$name")
    fi
}

skip_test() {
    printf "  ${SKIP_TAG}  %s (%s)\n" "$1" "$2"
    skip=$((skip + 1))
}

echo "== smoke =="
run "chimera --version"          "$CHIMERA" --version
run "chimera --help"             "$CHIMERA" --help
run "gen --help"                 "$CHIMERA" gen --help
run "chat --help"                "$CHIMERA" chat --help
run "tokenize --help"            "$CHIMERA" tokenize --help
run "embed --help"               "$CHIMERA" embed --help
run "whisper --help"             "$CHIMERA" whisper --help
run "sd --help"                  "$CHIMERA" sd --help

# CLI parse error must exit non-zero (missing required --model).
if "$CHIMERA" gen -p hi >/dev/null 2>&1; then
    printf "  ${FAIL_TAG}  %s\n" "gen without -m must exit non-zero"
    fail=$((fail + 1))
    failed_names+=("gen-missing-model")
else
    printf "  ${PASS_TAG}  %s\n" "gen without -m exits non-zero"
    pass=$((pass + 1))
fi

# Structured exit codes: 2 = BadInput (missing prompt + prompt-file).
GEN_MODEL_CHECK="$REPO_ROOT/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -f "$GEN_MODEL_CHECK" ]]; then
    "$CHIMERA" gen -m "$GEN_MODEL_CHECK" >/dev/null 2>&1
    code=$?
    if [[ $code -eq 2 ]]; then
        printf "  ${PASS_TAG}  %s\n" "gen without prompt exits 2 (BadInput)"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  %s (got %d, want 2)\n" "gen without prompt exits 2" "$code"
        fail=$((fail + 1))
        failed_names+=("gen-bad-input-exit")
    fi
fi
# Structured exit codes: 3 = Load (model not found).
"$CHIMERA" gen -m /no/such/model.gguf -p hi >/dev/null 2>&1
code=$?
if [[ $code -eq 3 ]]; then
    printf "  ${PASS_TAG}  %s\n" "gen with missing model exits 3 (Load)"
    pass=$((pass + 1))
else
    printf "  ${FAIL_TAG}  %s (got %d, want 3)\n" "gen missing-model exit" "$code"
    fail=$((fail + 1))
    failed_names+=("gen-load-exit")
fi

if [[ $SMOKE_ONLY -eq 1 ]]; then
    echo
    fail_color="$_GREEN"; [[ $fail -gt 0 ]] && fail_color="$_RED"
    printf "smoke-only: ${_GREEN}pass=%d${_RESET} ${fail_color}fail=%d${_RESET}\n" "$pass" "$fail"
    exit $fail
fi

echo
echo "== end-to-end =="

# gen: any small GGUF works. Prefer Llama-3.2-1B (smallest, fastest).
GEN_MODEL="$MODELS/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -f "$GEN_MODEL" ]]; then
    run "gen Llama-3.2-1B"        "$CHIMERA" gen -m "$GEN_MODEL" -p "Hello" -n 8
    # tokenize uses the same vocab; runs in seconds.
    run "tokenize Llama-3.2-1B"   "$CHIMERA" tokenize -m "$GEN_MODEL" -p "hello world"
    # prompt-file (stdin)
    if echo "Hi" | "$CHIMERA" gen -m "$GEN_MODEL" -f - -n 4 >/dev/null 2>&1; then
        printf "  ${PASS_TAG}  %s\n" "gen --prompt-file - (stdin)"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  %s\n" "gen --prompt-file - (stdin)"
        fail=$((fail + 1))
        failed_names+=("gen-prompt-file-stdin")
    fi
else
    skip_test "gen" "missing $GEN_MODEL"
fi

# embed: needs a true embedding model. Prefer bge-small / gte-small.
EMBED_MODEL=""
for candidate in bge-small-en-v1.5-q8_0.gguf gte-small-q8_0.gguf; do
    if [[ -f "$MODELS/$candidate" ]]; then
        EMBED_MODEL="$MODELS/$candidate"
        break
    fi
done
if [[ -n "$EMBED_MODEL" ]]; then
    run "embed $(basename "$EMBED_MODEL")" \
        "$CHIMERA" embed -m "$EMBED_MODEL" -p "a quick brown fox"

    # Embedding cache: two calls with the same input should produce
    # bit-identical output, and the second should not require the
    # embed model to recompute (we can't time-assert that here, but
    # the diff catches any cache-write-then-read corruption).
    EMB_CACHE_DB="$(mktemp -t chimera-embcache-XXXXXX).db"
    EMB_OUT1="$(mktemp -t chimera-emb1-XXXXXX)"
    EMB_OUT2="$(mktemp -t chimera-emb2-XXXXXX)"
    if "$CHIMERA" embed -m "$EMBED_MODEL" -p "cache me" \
            --cache-embeddings --cache-db "$EMB_CACHE_DB" > "$EMB_OUT1" 2>/dev/null \
       && "$CHIMERA" embed -m "$EMBED_MODEL" -p "cache me" \
            --cache-embeddings --cache-db "$EMB_CACHE_DB" > "$EMB_OUT2" 2>/dev/null \
       && diff -q "$EMB_OUT1" "$EMB_OUT2" >/dev/null; then
        printf "  ${PASS_TAG}  embedding cache (bit-identical round trip)\n"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  embedding cache (output differs or call failed)\n"
        fail=$((fail + 1))
        failed_names+=("embedding-cache")
    fi
    rm -f "$EMB_CACHE_DB" "$EMB_CACHE_DB-wal" "$EMB_CACHE_DB-shm" "$EMB_OUT1" "$EMB_OUT2"

    # Vector-store smoke: create a collection, ingest a small corpus
    # with one passage that should clearly win on a targeted query, and
    # check the top hit. Uses CHIMERA_DB so the test never touches the
    # user's real chimera.db. Cleaned up at the end of the block.
    VEC_DB="$(mktemp -t chimera-vec-XXXXXX).db"
    VEC_DOC="$(mktemp -t chimera-vec-doc-XXXXXX).txt"
    cat > "$VEC_DOC" <<'EOF'
The chimera serve subcommand exposes an OpenAI-compatible HTTP server.

The whisper.cpp wrapper accepts WAV audio files only in the first cut.

SQLite plus sqlite-vec is embedded for RAG and persistent chat history.
EOF
    if CHIMERA_DB="$VEC_DB" "$CHIMERA" index create -n rag_test -e "$EMBED_MODEL" \
            --chunk-tokens 128 --chunk-overlap 16 >/dev/null 2>&1 \
        && CHIMERA_DB="$VEC_DB" "$CHIMERA" index ingest -n rag_test -f "$VEC_DOC" \
            >/dev/null 2>&1; then
        VEC_HIT="$(CHIMERA_DB="$VEC_DB" "$CHIMERA" search -n rag_test \
            -q "audio file formats" -k 1 2>/dev/null)"
        if printf '%s' "$VEC_HIT" | grep -qi 'wav audio files'; then
            printf "  ${PASS_TAG}  %s\n" "vector store (rag_test top hit on 'audio file formats')"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s\n" "vector store top hit"
            printf "        got: %s\n" "$VEC_HIT"
            fail=$((fail + 1))
            failed_names+=("vector-store")
        fi
    else
        printf "  ${FAIL_TAG}  %s\n" "vector store create/ingest"
        fail=$((fail + 1))
        failed_names+=("vector-store-setup")
    fi
    rm -f "$VEC_DB" "$VEC_DB-wal" "$VEC_DB-shm" "$VEC_DOC"

    # Hybrid / lexical search: same setup as the semantic test above, but
    # the corpus carries a deliberately rare proper-noun-like token
    # ("xyzzyfrobnoc") that no semantic search would surface from a
    # generic query. Lexical mode must hit that chunk via BM25; hybrid
    # mode must produce output annotated with `rrf=` (the signal that
    # the RRF merge ran rather than a fall-through to one leg).
    HYB_DB="$(mktemp -t chimera-hyb-XXXXXX).db"
    HYB_DOC="$(mktemp -t chimera-hyb-doc-XXXXXX).txt"
    cat > "$HYB_DOC" <<'EOF'
Audio file formats commonly used in machine learning include WAV, FLAC, and MP3. WAV is uncompressed and preferred for training.

Image file formats vary widely. PNG is lossless. JPEG is lossy and not recommended for masks.

The xyzzyfrobnoc protocol is a fictitious wire format invented for this regression test. It carries no real semantic content but is keyword-unique within this corpus.

Photosynthesis converts light energy into chemical energy in plants.
EOF
    if CHIMERA_DB="$HYB_DB" "$CHIMERA" index create -n hyb_test -e "$EMBED_MODEL" \
            --chunk-tokens 96 --chunk-overlap 12 >/dev/null 2>&1 \
        && CHIMERA_DB="$HYB_DB" "$CHIMERA" index ingest -n hyb_test -f "$HYB_DOC" \
            >/dev/null 2>&1; then
        # Lexical leg must surface the rare token.
        LEX_OUT="$(CHIMERA_DB="$HYB_DB" "$CHIMERA" search -n hyb_test \
            -q "xyzzyfrobnoc" -k 1 --mode lexical 2>/dev/null || true)"
        if printf '%s' "$LEX_OUT" | grep -qi 'xyzzyfrobnoc'; then
            printf "  ${PASS_TAG}  %s\n" "search --mode lexical (BM25 hits rare token)"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s\n" "search --mode lexical"
            printf "        got: %s\n" "$LEX_OUT"
            fail=$((fail + 1))
            failed_names+=("search-lexical")
        fi

        # Hybrid leg: combined query with both a lexical anchor and a
        # semantic intent. Output format includes `rrf=<score>` only on
        # the hybrid path; verifying that field is present is a cheap
        # proof that the RRF merge actually ran.
        HYB_OUT="$(CHIMERA_DB="$HYB_DB" "$CHIMERA" search -n hyb_test \
            -q "xyzzyfrobnoc wire format" -k 3 --mode hybrid 2>/dev/null || true)"
        if printf '%s' "$HYB_OUT" | grep -q 'rrf=' \
            && printf '%s' "$HYB_OUT" | grep -qi 'xyzzyfrobnoc'; then
            printf "  ${PASS_TAG}  %s\n" "search --mode hybrid (RRF merge + rare-token recall)"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s\n" "search --mode hybrid"
            printf "        got: %s\n" "$HYB_OUT"
            fail=$((fail + 1))
            failed_names+=("search-hybrid")
        fi

        # Default mode is hybrid: omitting --mode should produce the
        # same `rrf=` annotation as --mode hybrid. Guards against a
        # future refactor accidentally flipping the default back.
        DEF_OUT="$(CHIMERA_DB="$HYB_DB" "$CHIMERA" search -n hyb_test \
            -q "xyzzyfrobnoc wire format" -k 3 2>/dev/null || true)"
        if printf '%s' "$DEF_OUT" | grep -q 'rrf='; then
            printf "  ${PASS_TAG}  %s\n" "search default mode is hybrid"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s\n" "search default mode"
            printf "        got: %s\n" "$DEF_OUT"
            fail=$((fail + 1))
            failed_names+=("search-default-mode")
        fi

        # Invalid --mode value: must exit non-zero (BadInput) without
        # crashing or producing hits. Guards the parse_search_mode entry
        # point on the CLI side.
        BAD_OUT="$(CHIMERA_DB="$HYB_DB" "$CHIMERA" search -n hyb_test \
            -q "anything" --mode nonsense 2>&1 || true)"
        if printf '%s' "$BAD_OUT" | grep -qi 'invalid --mode'; then
            printf "  ${PASS_TAG}  %s\n" "search --mode <bogus> exits with BadInput"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s\n" "search --mode validation"
            printf "        got: %s\n" "$BAD_OUT"
            fail=$((fail + 1))
            failed_names+=("search-mode-validation")
        fi
    else
        printf "  ${FAIL_TAG}  %s\n" "hybrid search create/ingest"
        fail=$((fail + 1))
        failed_names+=("hybrid-search-setup")
    fi
    rm -f "$HYB_DB" "$HYB_DB-wal" "$HYB_DB-shm" "$HYB_DOC"
else
    skip_test "embed" "no embedding model under $MODELS/"
    skip_test "vector store" "no embedding model under $MODELS/"
fi

# whisper: needs both a ggml model and a WAV. jfk.wav ships with whisper.cpp.
WHISPER_MODEL="$MODELS/ggml-base.en.bin"
WHISPER_WAV="$REPO_ROOT/build/whisper.cpp/samples/jfk.wav"
if [[ -f "$WHISPER_MODEL" && -f "$WHISPER_WAV" ]]; then
    run "whisper jfk.wav"         "$CHIMERA" whisper -m "$WHISPER_MODEL" -i "$WHISPER_WAV"
else
    if [[ ! -f "$WHISPER_MODEL" ]]; then
        skip_test "whisper" "missing $WHISPER_MODEL"
    else
        skip_test "whisper" "missing $WHISPER_WAV (run 'make deps' to fetch)"
    fi
fi

# sd: text-to-image. Smallest available diffusion model.
SD_MODEL=""
for candidate in sd_xl_turbo_1.0.q8_0.gguf v1-5-pruned-emaonly.q8_0.gguf z_image_turbo-Q6_K.gguf; do
    if [[ -f "$MODELS/$candidate" ]]; then
        SD_MODEL="$MODELS/$candidate"
        break
    fi
done
if [[ -n "$SD_MODEL" ]]; then
    SD_OUT="$(mktemp -t chimera-sd-XXXXXX).png"
    # Minimal step count + small image to keep the test fast.
    if "$CHIMERA" sd -m "$SD_MODEL" -p "a red cube" -o "$SD_OUT" \
        -W 256 -H 256 -s 2 >/dev/null 2>&1 && [[ -s "$SD_OUT" ]]; then
        printf "  ${PASS_TAG}  %s\n" "sd $(basename "$SD_MODEL")"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  %s\n" "sd $(basename "$SD_MODEL")"
        fail=$((fail + 1))
        failed_names+=("sd")
    fi
    # img2img: round-trip the just-produced image. Verifies the encode path
    # (vae_decode_only=false) and CLI plumbing for --init-image / --strength.
    if [[ -s "$SD_OUT" ]]; then
        SD_OUT2="$(mktemp -t chimera-sd2-XXXXXX).png"
        if "$CHIMERA" sd -m "$SD_MODEL" -p "a blue cube" \
            --init-image "$SD_OUT" --strength 0.6 \
            -o "$SD_OUT2" -W 256 -H 256 -s 2 >/dev/null 2>&1 \
            && [[ -s "$SD_OUT2" ]]; then
            printf "  ${PASS_TAG}  %s\n" "sd img2img round-trip"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s\n" "sd img2img round-trip"
            fail=$((fail + 1))
            failed_names+=("sd-img2img")
        fi
        rm -f "$SD_OUT2"
    fi
    rm -f "$SD_OUT"
else
    skip_test "sd" "no diffusion model under $MODELS/"
fi

# gen + mtmd: vision via mmproj. Needs a multimodal text model, its mmproj,
# and an input image. We synthesize the image on-the-fly via `sd`, then ask
# the VL model a question grounded in it. The assertion is loose -- the
# model must produce SOME output, indicating the mtmd pipeline (mmproj
# load -> bitmap load -> mtmd_tokenize -> mtmd_helper_eval_chunks ->
# sample_loop) ran end-to-end. Stricter content checks are unreliable on
# small/quantized VL models.
MTMD_TEXT_MODEL="$MODELS/gemma-4-E4B-it-Q4_K_M.gguf"
MTMD_MMPROJ="$MODELS/mmproj-gemma-4-E4B-it-BF16.gguf"
if [[ -f "$MTMD_TEXT_MODEL" && -f "$MTMD_MMPROJ" && -n "$SD_MODEL" ]]; then
    MTMD_IMG="$(mktemp -t chimera-vl-XXXXXX).png"
    if "$CHIMERA" sd -m "$SD_MODEL" -p "a single red apple, centered, white background" \
        -W 512 -H 512 -s 4 -o "$MTMD_IMG" >/dev/null 2>&1 && [[ -s "$MTMD_IMG" ]]; then
        VL_OUT="$("$CHIMERA" gen \
            -m "$MTMD_TEXT_MODEL" --mmproj "$MTMD_MMPROJ" --image "$MTMD_IMG" \
            -p "Describe this image briefly." -n 32 2>/dev/null || true)"
        if [[ -n "$(printf '%s' "$VL_OUT" | tr -d '[:space:]')" ]]; then
            printf "  ${PASS_TAG}  %s\n" "gen --mmproj --image (vision pipeline)"
            pass=$((pass + 1))
        else
            printf "  ${FAIL_TAG}  %s (empty output)\n" "gen --mmproj --image"
            fail=$((fail + 1))
            failed_names+=("mtmd")
        fi
    else
        skip_test "mtmd" "failed to synthesize input image"
    fi
    rm -f "$MTMD_IMG"
else
    skip_test "mtmd" "missing gemma model, mmproj, or sd model"
fi

# chat: verify the persistent KV cache propagates information across turns.
# Use a unique nonsense word that's very unlikely to appear by chance, so
# the test detects real cross-turn recall rather than a generic guess.
# Strip whitespace + lowercase the reply to tolerate the small model's
# tendency to render answers character-by-character on separate lines.
if [[ -f "$GEN_MODEL" ]]; then
    CHAT_REPLY="$(printf "my secret password is zephyrine.\nrepeat my secret password exactly.\n/exit\n" | \
        "$CHIMERA" chat -m "$GEN_MODEL" -n 32 2>/dev/null || true)"
    # collapse all whitespace, lowercase
    normalized="$(printf '%s' "$CHAT_REPLY" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"
    if printf '%s' "$normalized" | grep -q 'zephyrine'; then
        printf "  ${PASS_TAG}  %s\n" "chat persistent KV cache (recalls 'zephyrine')"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  %s\n" "chat persistent KV cache"
        printf "        reply: %s\n" "$CHAT_REPLY"
        fail=$((fail + 1))
        failed_names+=("chat-kv-cache")
    fi

    # chat --persist + --list + --search: verifies SQLite chat persistence
    # round-trips. Uses a unique nonsense word so the FTS5 hit is
    # unambiguous. The model's *reply* doesn't matter here — what we test
    # is that the conversation is saved and findable.
    CHAT_DB="$(mktemp -t chimera-chat-XXXXXX).db"
    PERSIST_OUT="$(printf "my secret token is xyzzy42.\n/exit\n" | \
        "$CHIMERA" chat -m "$GEN_MODEL" --db "$CHAT_DB" --persist -n 4 \
            --color never 2>/dev/null || true)"
    LIST_OUT="$("$CHIMERA"   chat --db "$CHAT_DB" --list             2>/dev/null || true)"
    SEARCH_OUT="$("$CHIMERA" chat --db "$CHAT_DB" --search xyzzy42   2>/dev/null || true)"
    if printf '%s' "$LIST_OUT"   | grep -q '#1' \
        && printf '%s' "$SEARCH_OUT" | grep -q '\[xyzzy42\]'; then
        printf "  ${PASS_TAG}  %s\n" "chat --persist + --list + --search round-trip"
        pass=$((pass + 1))
    else
        printf "  ${FAIL_TAG}  %s\n" "chat persistence round-trip"
        printf "        list:   %s\n" "$LIST_OUT"
        printf "        search: %s\n" "$SEARCH_OUT"
        fail=$((fail + 1))
        failed_names+=("chat-persistence")
    fi
    rm -f "$CHAT_DB" "$CHAT_DB-wal" "$CHAT_DB-shm"

    # X-Chimera-Chat-Id header: end-to-end against `chimera serve
    # --persist-chats`. Verifies four cases and the DB state they
    # produce. Skipped when `python3` or `curl` are missing — both ship
    # by default on every supported platform, but a stripped-down CI
    # container might not have them, and SKIP is friendlier than FAIL.
    if command -v python3 >/dev/null && command -v curl >/dev/null \
            && command -v sqlite3 >/dev/null; then
        HDR_DB="$(mktemp -t chimera-hdr-XXXXXX).db"
        HDR_LOG="$(mktemp -t chimera-hdr-log-XXXXXX)"
        HDR_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
        # Spawn serve on the chosen port. CPU-only (-gpu-layers 0) so the
        # test runs identically on machines without a GPU and doesn't
        # collide with concurrent GPU users.
        "$CHIMERA" serve -m "$GEN_MODEL" --persist-chats --chat-db "$HDR_DB" \
            --host 127.0.0.1 --port "$HDR_PORT" --gpu-layers 0 \
            > "$HDR_LOG" 2>&1 &
        HDR_PID=$!

        # Poll /health until 200 or 20 s elapsed.
        ready=0
        for _ in $(seq 1 40); do
            if curl -sS -o /dev/null -w '%{http_code}' \
                    "http://127.0.0.1:$HDR_PORT/health" 2>/dev/null \
                    | grep -q '^200$'; then
                ready=1
                break
            fi
            sleep 0.5
        done

        if [[ "$ready" != "1" ]]; then
            printf "  ${FAIL_TAG}  %s\n" "X-Chimera-Chat-Id (server failed to start)"
            printf "        log tail:\n"
            tail -5 "$HDR_LOG" | sed 's/^/          /'
            fail=$((fail + 1))
            failed_names+=("chat-id-header-startup")
        else
            # Case 1: no header on request → response header advertises
            # the new chat id.
            R1_HDR="$(curl -sS -D - -o /dev/null \
                -H 'Content-Type: application/json' \
                -d '{"model":"any","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":false}' \
                "http://127.0.0.1:$HDR_PORT/v1/chat/completions" 2>/dev/null)"
            FIRST_ID="$(printf '%s' "$R1_HDR" | grep -i '^X-Chimera-Chat-Id:' | tr -d '\r' | awk '{print $2}')"
            if [[ -n "$FIRST_ID" ]]; then
                printf "  ${PASS_TAG}  %s\n" "X-Chimera-Chat-Id new chat → header echoed"
                pass=$((pass + 1))
            else
                printf "  ${FAIL_TAG}  %s\n" "X-Chimera-Chat-Id new chat"
                printf "        got headers: %s\n" "$R1_HDR"
                fail=$((fail + 1))
                failed_names+=("chat-id-header-new")
            fi

            # Case 2: echo the captured id on a follow-up request →
            # response must carry the same id and the DB must add only
            # the new user + assistant rows (not duplicate the prior
            # turn).
            if [[ -n "$FIRST_ID" ]]; then
                R2_HDR="$(curl -sS -D - -o /dev/null \
                    -H 'Content-Type: application/json' \
                    -H "X-Chimera-Chat-Id: $FIRST_ID" \
                    -d '{"model":"any","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"again"}],"max_tokens":4,"stream":false}' \
                    "http://127.0.0.1:$HDR_PORT/v1/chat/completions" 2>/dev/null)"
                R2_ID="$(printf '%s' "$R2_HDR" | grep -i '^X-Chimera-Chat-Id:' | tr -d '\r' | awk '{print $2}')"
                if [[ "$R2_ID" == "$FIRST_ID" ]]; then
                    printf "  ${PASS_TAG}  %s\n" "X-Chimera-Chat-Id echo → same id reused"
                    pass=$((pass + 1))
                else
                    printf "  ${FAIL_TAG}  %s (expected %s, got %s)\n" \
                        "X-Chimera-Chat-Id echo" "$FIRST_ID" "$R2_ID"
                    fail=$((fail + 1))
                    failed_names+=("chat-id-header-echo")
                fi
            fi

            # Case 3: unknown id → HTTP 404, no inner handler invocation
            # (verified indirectly: the chat row count in the DB does
            # not change).
            R3_CODE="$(curl -sS -o /dev/null -w '%{http_code}' \
                -H 'Content-Type: application/json' \
                -H 'X-Chimera-Chat-Id: 9999999' \
                -d '{"model":"any","messages":[{"role":"user","content":"x"}],"max_tokens":4,"stream":false}' \
                "http://127.0.0.1:$HDR_PORT/v1/chat/completions" 2>/dev/null || true)"
            if [[ "$R3_CODE" == "404" ]]; then
                printf "  ${PASS_TAG}  %s\n" "X-Chimera-Chat-Id unknown id → 404"
                pass=$((pass + 1))
            else
                printf "  ${FAIL_TAG}  %s (got HTTP %s)\n" \
                    "X-Chimera-Chat-Id unknown id" "$R3_CODE"
                fail=$((fail + 1))
                failed_names+=("chat-id-header-unknown")
            fi

            # Case 4: malformed id (non-integer) → HTTP 400.
            R4_CODE="$(curl -sS -o /dev/null -w '%{http_code}' \
                -H 'Content-Type: application/json' \
                -H 'X-Chimera-Chat-Id: notanint' \
                -d '{"model":"any","messages":[{"role":"user","content":"x"}],"max_tokens":4,"stream":false}' \
                "http://127.0.0.1:$HDR_PORT/v1/chat/completions" 2>/dev/null || true)"
            if [[ "$R4_CODE" == "400" ]]; then
                printf "  ${PASS_TAG}  %s\n" "X-Chimera-Chat-Id malformed → 400"
                pass=$((pass + 1))
            else
                printf "  ${FAIL_TAG}  %s (got HTTP %s)\n" \
                    "X-Chimera-Chat-Id malformed" "$R4_CODE"
                fail=$((fail + 1))
                failed_names+=("chat-id-header-malformed")
            fi
        fi

        # Tear down before inspecting the DB so WAL is checkpointed via
        # the shutdown path. SIGTERM, then escalate to KILL if it lingers.
        if kill -0 "$HDR_PID" 2>/dev/null; then
            kill -TERM "$HDR_PID" 2>/dev/null || true
            for _ in $(seq 1 20); do
                kill -0 "$HDR_PID" 2>/dev/null || break
                sleep 0.25
            done
            kill -KILL "$HDR_PID" 2>/dev/null || true
        fi
        wait "$HDR_PID" 2>/dev/null || true

        # DB-state assertion. After cases 1 and 2 we expect exactly one
        # chats row (id reused via the header). Cases 3 and 4 must NOT
        # have created rows. Messages: 2 from case 1 (user+assistant) +
        # 2 from case 2 (last user "again" + assistant) = 4. The
        # malformed/unknown cases are negative tests so their rejection
        # propagating to "no rows" is the assertion.
        if [[ "$ready" == "1" ]]; then
            ROW_COUNTS="$(sqlite3 "$HDR_DB" \
                "SELECT (SELECT COUNT(*) FROM chats) || ' ' || (SELECT COUNT(*) FROM messages)" \
                2>/dev/null || echo "ERR")"
            if [[ "$ROW_COUNTS" == "1 4" ]]; then
                printf "  ${PASS_TAG}  %s\n" "X-Chimera-Chat-Id DB state (1 chat, 4 messages)"
                pass=$((pass + 1))
            else
                printf "  ${FAIL_TAG}  %s (got '%s', expected '1 4')\n" \
                    "X-Chimera-Chat-Id DB state" "$ROW_COUNTS"
                fail=$((fail + 1))
                failed_names+=("chat-id-header-db-state")
            fi
        fi

        rm -f "$HDR_DB" "$HDR_DB-wal" "$HDR_DB-shm" "$HDR_LOG"
    else
        skip_test "X-Chimera-Chat-Id" "needs python3 + curl + sqlite3"
    fi
fi

echo
# Color the fail-count green at zero, red otherwise; pass is always
# green if non-zero. The summary line is what readers scan first, so
# the at-a-glance signal lives here.
fail_color="$_GREEN"; [[ $fail -gt 0 ]] && fail_color="$_RED"
printf "summary: ${_GREEN}pass=%d${_RESET} ${fail_color}fail=%d${_RESET} ${_YELLOW}skip=%d${_RESET}\n" \
    "$pass" "$fail" "$skip"
if [[ $fail -gt 0 ]]; then
    printf "${_RED}failed:${_RESET} %s\n" "${failed_names[*]}"
fi
exit $fail

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
    echo "FAIL: chimera binary not found under $REPO_ROOT/build/" >&2
    echo "      searched: build/chimera, build/chimera.exe, build/{Release,Debug}/chimera[.exe]" >&2
    echo "      (run 'make build' first, or set CHIMERA=...)" >&2
    exit 1
fi

pass=0
fail=0
skip=0
failed_names=()

run() {
    local name="$1"; shift
    if "$@" >/dev/null 2>&1; then
        printf "  PASS  %s\n" "$name"
        pass=$((pass + 1))
    else
        printf "  FAIL  %s\n" "$name"
        fail=$((fail + 1))
        failed_names+=("$name")
    fi
}

skip_test() {
    printf "  SKIP  %s (%s)\n" "$1" "$2"
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
    printf "  FAIL  %s\n" "gen without -m must exit non-zero"
    fail=$((fail + 1))
    failed_names+=("gen-missing-model")
else
    printf "  PASS  %s\n" "gen without -m exits non-zero"
    pass=$((pass + 1))
fi

# Structured exit codes: 2 = BadInput (missing prompt + prompt-file).
GEN_MODEL_CHECK="$REPO_ROOT/models/Llama-3.2-1B-Instruct-Q8_0.gguf"
if [[ -f "$GEN_MODEL_CHECK" ]]; then
    "$CHIMERA" gen -m "$GEN_MODEL_CHECK" >/dev/null 2>&1
    code=$?
    if [[ $code -eq 2 ]]; then
        printf "  PASS  %s\n" "gen without prompt exits 2 (BadInput)"
        pass=$((pass + 1))
    else
        printf "  FAIL  %s (got %d, want 2)\n" "gen without prompt exits 2" "$code"
        fail=$((fail + 1))
        failed_names+=("gen-bad-input-exit")
    fi
fi
# Structured exit codes: 3 = Load (model not found).
"$CHIMERA" gen -m /no/such/model.gguf -p hi >/dev/null 2>&1
code=$?
if [[ $code -eq 3 ]]; then
    printf "  PASS  %s\n" "gen with missing model exits 3 (Load)"
    pass=$((pass + 1))
else
    printf "  FAIL  %s (got %d, want 3)\n" "gen missing-model exit" "$code"
    fail=$((fail + 1))
    failed_names+=("gen-load-exit")
fi

if [[ $SMOKE_ONLY -eq 1 ]]; then
    echo
    echo "smoke-only: pass=$pass fail=$fail"
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
        printf "  PASS  %s\n" "gen --prompt-file - (stdin)"
        pass=$((pass + 1))
    else
        printf "  FAIL  %s\n" "gen --prompt-file - (stdin)"
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
        printf "  PASS  embedding cache (bit-identical round trip)\n"
        pass=$((pass + 1))
    else
        printf "  FAIL  embedding cache (output differs or call failed)\n"
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
            printf "  PASS  %s\n" "vector store (rag_test top hit on 'audio file formats')"
            pass=$((pass + 1))
        else
            printf "  FAIL  %s\n" "vector store top hit"
            printf "        got: %s\n" "$VEC_HIT"
            fail=$((fail + 1))
            failed_names+=("vector-store")
        fi
    else
        printf "  FAIL  %s\n" "vector store create/ingest"
        fail=$((fail + 1))
        failed_names+=("vector-store-setup")
    fi
    rm -f "$VEC_DB" "$VEC_DB-wal" "$VEC_DB-shm" "$VEC_DOC"
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
        printf "  PASS  %s\n" "sd $(basename "$SD_MODEL")"
        pass=$((pass + 1))
    else
        printf "  FAIL  %s\n" "sd $(basename "$SD_MODEL")"
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
            printf "  PASS  %s\n" "sd img2img round-trip"
            pass=$((pass + 1))
        else
            printf "  FAIL  %s\n" "sd img2img round-trip"
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
            printf "  PASS  %s\n" "gen --mmproj --image (vision pipeline)"
            pass=$((pass + 1))
        else
            printf "  FAIL  %s (empty output)\n" "gen --mmproj --image"
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
        printf "  PASS  %s\n" "chat persistent KV cache (recalls 'zephyrine')"
        pass=$((pass + 1))
    else
        printf "  FAIL  %s\n" "chat persistent KV cache"
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
        printf "  PASS  %s\n" "chat --persist + --list + --search round-trip"
        pass=$((pass + 1))
    else
        printf "  FAIL  %s\n" "chat persistence round-trip"
        printf "        list:   %s\n" "$LIST_OUT"
        printf "        search: %s\n" "$SEARCH_OUT"
        fail=$((fail + 1))
        failed_names+=("chat-persistence")
    fi
    rm -f "$CHAT_DB" "$CHAT_DB-wal" "$CHAT_DB-shm"
fi

echo
echo "summary: pass=$pass fail=$fail skip=$skip"
if [[ $fail -gt 0 ]]; then
    printf "failed: %s\n" "${failed_names[*]}"
fi
exit $fail

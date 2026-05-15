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
CHIMERA="${CHIMERA:-$REPO_ROOT/build/chimera}"
MODELS="$REPO_ROOT/models"
SMOKE_ONLY=0
[[ "${1:-}" == "--smoke" ]] && SMOKE_ONLY=1

if [[ ! -x "$CHIMERA" ]]; then
    echo "FAIL: chimera binary not found at $CHIMERA" >&2
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
else
    skip_test "gen" "missing $GEN_MODEL"
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
    rm -f "$SD_OUT"
else
    skip_test "sd" "no diffusion model under $MODELS/"
fi

echo
echo "summary: pass=$pass fail=$fail skip=$skip"
if [[ $fail -gt 0 ]]; then
    printf "failed: %s\n" "${failed_names[*]}"
fi
exit $fail

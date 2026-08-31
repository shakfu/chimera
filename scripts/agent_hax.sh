#!/usr/bin/env sh
# agent_hax.sh -- run the hax coding agent against a chimera-hosted model.
#
#   scripts/agent_hax.sh MODEL.gguf [hax args...]
#
# hax (https://github.com/shakfu/hax) is an external binary: nothing here is
# vendored, linked, or pinned. The two connect over loopback HTTP because
# `chimera serve` already speaks the OpenAI Chat Completions API that hax's
# `llama.cpp` provider expects, jinja templating (and therefore tool calling)
# is on by default, and ServeOptions.n_ctx=0 gives the model's full training
# context rather than a truncated default.
#
# This script exists to make the pairing one command: pick a free port, start
# the server, block until it answers /health, point hax at it, and guarantee
# the server dies when hax does.
#
# Environment:
#   CHIMERA             chimera binary            (default: <repo>/build/chimera)
#   HAX                 hax binary                (default: hax on PATH)
#   CHIMERA_AGENT_PORT  first port to try         (default: 8080)
#   CHIMERA_SERVE_ARGS  extra `chimera serve` flags, word-split on purpose
#                       (e.g. "--gpu-layers 99 -c 32768")
#   CHIMERA_AGENT_TIMEOUT  seconds to wait for model load (default: 180)
#
# Serving with `--api-key K` additionally needs HAX_LLAMACPP_API_KEY=K.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

die() { printf 'agent_hax.sh: %s\n' "$*" >&2; exit 1; }

case "${1-}" in
    -h|--help|'')
        sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

model=$1
shift

[ -f "$model" ] || die "no such model file: $model"

chimera=${CHIMERA:-$root/build/chimera}
[ -x "$chimera" ] || chimera=$(command -v chimera 2>/dev/null) \
    || die "chimera binary not found; run 'make build' or set CHIMERA=<path>"

hax=${HAX:-hax}
command -v "$hax" >/dev/null 2>&1 \
    || die "hax not found on PATH; install it or set HAX=<path/to/hax>"

# curl exit 7 is "failed to connect", i.e. nothing is listening. Any other
# outcome means the port is taken -- by another chimera, or by anything else.
port_free() {
    rc=0
    curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$1/health" || rc=$?
    [ "$rc" -eq 7 ]
}

port=${CHIMERA_AGENT_PORT:-8080}
tries=0
while ! port_free "$port"; do
    tries=$((tries + 1))
    [ "$tries" -lt 20 ] || die "no free port in $((port - tries))..$port"
    port=$((port + 1))
done

# `trap '' INT` before exec makes the server ignore terminal SIGINT, which an
# unignored background child in this process group would otherwise receive:
# Ctrl-C in hax interrupts a turn, and must not take the model server with it.
# SIG_IGN survives exec; a handler would not. SIGTERM stays default so the
# cleanup below still works.
# shellcheck disable=SC2086  # CHIMERA_SERVE_ARGS is split deliberately
( trap '' INT; exec "$chimera" serve -m "$model" --port "$port" ${CHIMERA_SERVE_ARGS:-} ) &
srv=$!
trap 'kill "$srv" 2>/dev/null || true' EXIT TERM

# Absorb Ctrl-C here rather than dying: hax owns the terminal and handles it.
# A handler (not '') is required -- children inherit SIG_IGN across exec, which
# would leave hax unable to interrupt its own turn.
trap : INT

timeout=${CHIMERA_AGENT_TIMEOUT:-180}
ticks=0
until curl -sf -o /dev/null "http://127.0.0.1:$port/health"; do
    kill -0 "$srv" 2>/dev/null || die "chimera serve exited before becoming ready"
    ticks=$((ticks + 1))
    [ "$ticks" -lt $((timeout * 5)) ] || die "server not ready after ${timeout}s"
    sleep 0.2
done

# hax adopts the single served model from /v1/models, so HAX_MODEL is not
# needed. The catalog fetch is disabled by default because models.dev has no
# entry for a local GGUF and hax reads context size from /props anyway; an
# explicit HAX_CATALOG_URL in the environment still wins.
HAX_PROVIDER=llama.cpp \
HAX_LLAMACPP_BASE_URL="http://127.0.0.1:$port/v1" \
HAX_CATALOG_URL="${HAX_CATALOG_URL-}" \
    "$hax" "$@" || status=$?

exit "${status:-0}"

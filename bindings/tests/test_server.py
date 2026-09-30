"""``Server.run()`` end to end: start it, make requests, stop it.

``run()`` blocks and installs process-wide signal handlers, so it runs in a
child interpreter. The child exits with ``run()``'s return value. SKIPs when the
llama model is absent; see ``conftest.py``.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ALIAS = "bindings-test"
STARTUP_TIMEOUT_S = 120
SHUTDOWN_TIMEOUT_S = 30

CHILD = """
import sys, chimera
opts = chimera.ServeOptions()
opts.model, opts.port, opts.alias = sys.argv[1], int(sys.argv[2]), sys.argv[3]
opts.gpu_layers = 0
opts.n_ctx = 512
sys.exit(chimera.Server(opts).run())
"""


# Runs the server twice: stopped by SIGINT, then by the shutdown route with a
# SIGINT inside its 150 ms delay. Then sends a last SIGINT and exits 0 only if
# that one raises KeyboardInterrupt.
CHILD_SIGNALS = """
import os, signal, sys, threading, time, urllib.request, chimera
opts = chimera.ServeOptions()
opts.model, opts.port = sys.argv[1], int(sys.argv[2])
opts.gpu_layers = 0
opts.n_ctx = 512
base = f"http://127.0.0.1:{opts.port}"

def stop_when_healthy(post_shutdown):
    while True:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5):
                break
        except OSError:
            time.sleep(0.2)
    time.sleep(0.5)  # /health turns 200 just before run() installs its handlers
    if post_shutdown:
        urllib.request.urlopen(f"{base}/v1/chimera/shutdown", data=b"{}", timeout=5).close()
    os.kill(os.getpid(), signal.SIGINT)

srv = chimera.Server(opts)
for post_shutdown in (False, True):
    t = threading.Thread(target=stop_when_healthy, args=(post_shutdown,))
    t.start()
    if srv.run() != 0:
        sys.exit(3)
    t.join()
time.sleep(0.5)  # outlive the shutdown delay
try:
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(5)
except KeyboardInterrupt:
    sys.exit(0)
sys.exit(4)
"""

# handle_signals=False: stop() ends a run during startup, then a second run
# leaves SIGINT to Python and ends on stop(). Exits 0 only if both hold.
CHILD_STOP = """
import os, signal, sys, threading, time, urllib.request, chimera
opts = chimera.ServeOptions()
opts.model, opts.port = sys.argv[1], int(sys.argv[2])
opts.gpu_layers = 0
opts.n_ctx = 512
opts.handle_signals = False
health = f"http://127.0.0.1:{opts.port}/health"

def healthy():
    try:
        with urllib.request.urlopen(health, timeout=5):
            return True
    except OSError:
        return False

srv = chimera.Server(opts)
srv.stop()  # no run() in progress: no-op
rc = []

t = threading.Thread(target=lambda: rc.append(srv.run()))
t.start()
time.sleep(0.2)  # run() has started; the model is still loading or just loaded
srv.stop()
t.join()
if rc != [0]:
    sys.exit(3)

t = threading.Thread(target=lambda: rc.append(srv.run()))
t.start()
while not healthy():
    time.sleep(0.2)
try:
    os.kill(os.getpid(), signal.SIGINT)
    time.sleep(5)
    sys.exit(4)  # chimera took the signal
except KeyboardInterrupt:
    pass
if not healthy():
    sys.exit(5)  # the signal stopped the server
srv.stop()
t.join()
sys.exit(0 if rc == [0, 0] else 6)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(chimera_mod, script: str, *argv: str) -> subprocess.Popen:
    # The child must import the same module this session did.
    env = {**os.environ, "PYTHONPATH": str(Path(chimera_mod.__file__).parent)}
    return subprocess.Popen(
        [sys.executable, "-c", script, *argv],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _request(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {}


@pytest.fixture(scope="module")
def server(chimera_mod, llama_model):
    """(base url, process) for a running server; killed at teardown if still up."""
    port = _free_port()
    proc = _spawn(chimera_mod, CHILD, str(llama_model), str(port), ALIAS)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    try:
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited during startup (rc={proc.returncode})")
            if time.monotonic() > deadline:
                pytest.fail(f"server not healthy after {STARTUP_TIMEOUT_S}s")
            try:
                # /health answers 503 until the model has loaded.
                if _request(f"{base}/health")[0] == 200:
                    break
            except OSError:
                pass  # not listening yet
            time.sleep(0.2)
        yield base, proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_models_reports_alias(server):
    base, _ = server
    status, body = _request(f"{base}/v1/models")
    assert status == 200
    assert [m["id"] for m in body["data"]] == [ALIAS]


def test_chat_completion(server):
    base, _ = server
    status, body = _request(
        f"{base}/v1/chat/completions",
        {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4},
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"]


def test_native_completion(server):
    base, _ = server
    status, body = _request(f"{base}/completion", {"prompt": "Hello", "n_predict": 2})
    assert status == 200
    assert body["content"]


# Last: it stops the module's server.
def test_run_returns_zero_on_shutdown(server):
    base, proc = server
    status, _ = _request(f"{base}/v1/chimera/shutdown", {})
    assert status == 202
    assert proc.wait(timeout=SHUTDOWN_TIMEOUT_S) == 0


def _child_exit_code(chimera_mod, script: str, llama_model: Path) -> int:
    proc = _spawn(chimera_mod, script, str(llama_model), str(_free_port()))
    try:
        return proc.wait(timeout=STARTUP_TIMEOUT_S)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill cannot deliver SIGINT on Windows")
def test_run_restores_signal_handlers(chimera_mod, llama_model):
    # run() used to leave its SIGINT handler installed, pointing at a closure
    # over its own dead locals, and the second-interrupt flag set.
    assert _child_exit_code(chimera_mod, CHILD_SIGNALS, llama_model) == 0


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill cannot deliver SIGINT on Windows")
def test_stop_without_signal_handlers(chimera_mod, llama_model):
    assert _child_exit_code(chimera_mod, CHILD_STOP, llama_model) == 0

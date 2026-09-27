"""scripts/regenerate_evidence.py stops before ANY step when Phoenix is unreachable or its OTLP endpoint does not
accept spans. Offline: a free local port (nothing listening) and a tiny local stub server stand in for Phoenix."""
from __future__ import annotations

import importlib.util
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src import config
from src.observability import tracing

ROOT = Path(__file__).resolve().parent.parent


def _load_regen():
    spec = importlib.util.spec_from_file_location("regenerate_evidence", ROOT / "scripts" / "regenerate_evidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def regen(monkeypatch):
    """The script module with every step recorded instead of run, and Phoenix launching disabled."""
    mod = _load_regen()
    ran: list[str] = []
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: ran.append("subprocess"))
    monkeypatch.setattr(mod, "reset_state", lambda: ran.append("reset_state"))
    monkeypatch.setattr(mod, "run_all", lambda args: ran.append("run_all"))
    monkeypatch.setattr(mod.asyncio, "run", lambda coro: ran.append("asyncio.run"))
    launches: list[int] = []
    monkeypatch.setattr(tracing, "_start_phoenix", lambda: launches.append(config.PHOENIX_PORT))
    monkeypatch.setattr(sys, "argv", ["regenerate_evidence.py"])
    return mod, ran, launches


class _StubPhoenix(BaseHTTPRequestHandler):
    traces_status = 200

    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.end_headers()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(type(self).traces_status if self.path == "/v1/traces" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_phoenix(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _StubPhoenix)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "PHOENIX_PORT", server.server_address[1])
    yield _StubPhoenix
    server.shutdown()
    server.server_close()
    _StubPhoenix.traces_status = 200


def test_unreachable_phoenix_stops_before_any_step(regen, monkeypatch, capsys):
    mod, ran, launches = regen
    monkeypatch.setattr(config, "PHOENIX_PORT", _free_port())
    with pytest.raises(SystemExit) as exc:
        mod.main()
    msg = str(exc.value)
    assert "PREFLIGHT FAILED" in msg and f"localhost:{config.PHOENIX_PORT}/healthz" in msg
    assert "Nothing was run" in msg
    assert launches == [config.PHOENIX_PORT]  # it tried to launch Phoenix, as step 3 would
    assert ran == []  # no synthetic data, no reset, no demo/eval/tests
    assert "[1/10]" not in capsys.readouterr().out


def test_otlp_endpoint_rejecting_spans_stops_before_any_step(regen, stub_phoenix):
    mod, ran, launches = regen
    stub_phoenix.traces_status = 415
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert "OTLP endpoint does not accept spans" in str(exc.value) and "415" in str(exc.value)
    assert ran == [] and launches == []  # Phoenix was already up, so no launch either


def test_preflight_passes_when_phoenix_accepts_spans(regen, stub_phoenix, capsys):
    mod, ran, _ = regen
    mod.preflight_phoenix()
    assert "OTLP endpoint accepts spans" in capsys.readouterr().out
    assert ran == []

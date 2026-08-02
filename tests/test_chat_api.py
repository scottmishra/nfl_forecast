"""`/api/chat` — SSE framing, session lifecycle, and graceful unavailability.

No model and no `claude` subprocess: the SDK client is stubbed, so these run
anywhere and stay fast. What's under test is the plumbing the agent rides on —
that a tool call reaches the panel as a chip, that a dead turn doesn't poison
the next question, and that a missing SDK degrades instead of 500ing.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from gameday.agent import session as session_mod
from gameday.api import chat
from gameday.api.server import app

pytest.importorskip("claude_agent_sdk", reason="needs the [agent] extra")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock  # noqa: E402

client = TestClient(app)


# --------------------------------------------------------------------------
# a stand-in for ClaudeSDKClient
# --------------------------------------------------------------------------

def _assistant(*blocks):
    return AssistantMessage(content=list(blocks), model="stub")


def _result(**kw):
    defaults = dict(subtype="success", duration_ms=1, duration_api_ms=1,
                    is_error=False, num_turns=2, session_id="s",
                    total_cost_usd=0.01, result=None)
    defaults.update(kw)
    return ResultMessage(**defaults)


class StubClient:
    """Replays a scripted message sequence, or raises."""

    def __init__(self, script=None, raises=None):
        self.script = script or []
        self.raises = raises
        self.queries: list[str] = []
        self.disconnected = False

    async def connect(self):
        return None

    async def query(self, prompt, session_id="default"):
        self.queries.append(prompt)

    async def receive_response(self):
        if self.raises:
            raise self.raises
        for message in self.script:
            yield message

    async def disconnect(self):
        self.disconnected = True


@pytest.fixture
def manager(monkeypatch):
    """A fresh SessionManager whose clients are stubs."""
    mgr = session_mod.SessionManager(max_sessions=2, ttl_minutes=30)
    made: list[StubClient] = []

    script = [
        _assistant(ToolUseBlock(id="t1", name="mcp__gameday__get_draft_board", input={})),
        _assistant(TextBlock(text="Take him. ")),
        _assistant(TextBlock(text="VORP +95.")),
        _result(),
    ]

    async def fake_get(session_id):
        existing = mgr._sessions.get(session_id)
        if existing:
            return existing
        stub = StubClient(script=list(script))
        made.append(stub)
        sess = session_mod.Session(session_id=session_id, client=stub)
        mgr._sessions[session_id] = sess
        return sess

    monkeypatch.setattr(mgr, "get", fake_get)
    monkeypatch.setattr(chat, "MANAGER", mgr)
    monkeypatch.setattr(chat, "availability",
                        lambda: {"available": True, "sdk_installed": True,
                                 "cli_path": "/stub/claude", "auth_source": "stub",
                                 "model": "stub", "problems": []})
    mgr.made = made
    return mgr


def _events(response):
    """Parse an SSE body into [(event, data), ...]."""
    out = []
    for frame in response.text.split("\n\n"):
        if not frame.strip():
            continue
        name = data = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name:
            out.append((name, data))
    return out


def _ask(question="who?", session_id="s1", context=None):
    return client.post("/api/chat", json={
        "session_id": session_id, "message": question, "context": context})


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------

def test_stream_carries_tool_then_text_then_done(manager):
    events = _events(_ask())
    assert [name for name, _ in events] == ["tool", "text", "text", "done"]


def test_tool_events_carry_a_human_label(manager):
    name, payload = _events(_ask())[0]
    assert name == "tool"
    assert payload["name"] == "mcp__gameday__get_draft_board"
    assert payload["label"] == "reading the draft board"


def test_unknown_tool_still_gets_a_readable_label():
    assert chat._tool_label("mcp__gameday__some_new_thing") == "using some new thing"
    assert chat._tool_label("WebSearch") == "searching the news"


def test_text_arrives_in_order_for_incremental_render(manager):
    texts = [p["text"] for n, p in _events(_ask()) if n == "text"]
    assert "".join(texts) == "Take him. VORP +95."


def test_done_reports_turns_and_cost(manager):
    _, payload = _events(_ask())[-1]
    assert payload["turns"] == 2 and payload["is_error"] is False
    assert payload["cost_usd"] == pytest.approx(0.01)


def test_response_is_an_unbuffered_event_stream(manager):
    """Proxy buffering would hold the whole answer until the turn ended."""
    res = _ask()
    assert res.headers["content-type"].startswith("text/event-stream")
    assert res.headers["cache-control"] == "no-cache"
    assert res.headers["x-accel-buffering"] == "no"


# --------------------------------------------------------------------------
# context and sessions
# --------------------------------------------------------------------------

def test_current_view_is_attached_to_the_question(manager):
    """So "what about him?" resolves without retyping a name."""
    _ask("what about him?", context="Draft board, RB only")
    sent = manager._sessions["s1"].client.queries[0]
    assert "Draft board, RB only" in sent and "what about him?" in sent


def test_no_context_leaves_the_question_alone(manager):
    _ask("plain question")
    assert manager._sessions["s1"].client.queries[0] == "plain question"


def test_conversation_is_reused_across_turns(manager):
    _ask(session_id="same")
    _ask(session_id="same")
    assert len(manager.made) == 1                       # one client, two turns
    assert len(manager._sessions["same"].client.queries) == 2


def test_reset_drops_the_conversation(manager):
    _ask(session_id="doomed")
    stub = manager._sessions["doomed"].client
    body = client.delete("/api/chat/doomed").json()
    assert body["dropped"] is True and stub.disconnected
    assert client.delete("/api/chat/doomed").json()["dropped"] is False


# --------------------------------------------------------------------------
# failure paths
# --------------------------------------------------------------------------

def test_a_failed_turn_does_not_poison_the_conversation(manager, monkeypatch):
    """The dead client is dropped, so the next question starts clean instead of
    failing forever against a broken subprocess."""
    async def failing_get(session_id):
        sess = session_mod.Session(
            session_id=session_id,
            client=StubClient(raises=RuntimeError("subprocess died")))
        manager._sessions[session_id] = sess
        return sess

    monkeypatch.setattr(manager, "get", failing_get)
    events = _events(_ask(session_id="broken"))
    assert events[-1][0] == "status"
    assert "subprocess died" in events[-1][1]["message"]
    assert "broken" not in manager._sessions


def test_expired_token_surfaces_as_an_api_error(manager, monkeypatch):
    """A 401 comes back as a successful stream with is_error set — if this were
    swallowed the panel would show an empty answer and no reason."""
    async def get(session_id):
        sess = session_mod.Session(session_id=session_id, client=StubClient(script=[
            _result(is_error=True, api_error_status=401,
                    result="Failed to authenticate. API Error: 401")]))
        manager._sessions[session_id] = sess
        return sess

    monkeypatch.setattr(manager, "get", get)
    name, payload = _events(_ask(session_id="expired"))[-1]
    assert name == "done" and payload["is_error"] is True
    assert payload["api_error_status"] == 401


def test_unavailable_agent_returns_503_with_the_diagnosis(monkeypatch):
    """The dashboard must keep working; the panel needs to say what's missing."""
    monkeypatch.setattr(chat, "availability", lambda: {
        "available": False, "sdk_installed": False, "cli_path": None,
        "auth_source": None, "model": "stub",
        "problems": ["claude-agent-sdk is not installed"]})
    res = _ask()
    assert res.status_code == 503
    assert "claude-agent-sdk is not installed" in res.json()["detail"]["problems"]


def test_health_reports_what_is_missing():
    body = client.get("/api/chat/health").json()
    assert {"available", "sdk_installed", "cli_path",
            "auth_source", "problems"} <= set(body)


@pytest.mark.parametrize("payload", [
    {"session_id": "", "message": "hi"},          # no session
    {"session_id": "s", "message": ""},           # empty question
    {"session_id": "s", "message": "x" * 5000},   # oversized
])
def test_malformed_requests_are_rejected(payload):
    assert client.post("/api/chat", json=payload).status_code == 422


# --------------------------------------------------------------------------
# session manager bookkeeping
# --------------------------------------------------------------------------

def test_idle_sessions_are_evicted():
    mgr = session_mod.SessionManager(max_sessions=5, ttl_minutes=0.0)
    stub = StubClient()
    mgr._sessions["old"] = session_mod.Session(session_id="old", client=stub)
    mgr._sessions["old"].last_used -= 600  # ten minutes ago
    asyncio.run(mgr._evict_idle())
    assert "old" not in mgr._sessions and stub.disconnected


def test_shutdown_closes_every_client():
    mgr = session_mod.SessionManager()
    stubs = [StubClient(), StubClient()]
    for i, stub in enumerate(stubs):
        mgr._sessions[str(i)] = session_mod.Session(session_id=str(i), client=stub)
    asyncio.run(mgr.shutdown())
    assert mgr.live == 0 and all(s.disconnected for s in stubs)


def test_teardown_survives_a_client_that_fails_to_close():
    class Stubborn(StubClient):
        async def disconnect(self):
            raise RuntimeError("already gone")

    mgr = session_mod.SessionManager()
    mgr._sessions["x"] = session_mod.Session(session_id="x", client=Stubborn())
    asyncio.run(mgr.shutdown())   # must not raise
    assert mgr.live == 0


# --------------------------------------------------------------------------
# environment resolution
# --------------------------------------------------------------------------

def test_auth_source_prefers_the_long_lived_token(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    assert session_mod.auth_source() == "CLAUDE_CODE_OAUTH_TOKEN"


def test_auth_source_falls_back_to_the_api_key(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    assert session_mod.auth_source() == "ANTHROPIC_API_KEY"


def test_explicit_cli_path_is_honoured(monkeypatch, tmp_path):
    """systemd user units don't inherit ~/.local/bin, so this override is the
    escape hatch when the CLI isn't where `which` would find it."""
    fake = tmp_path / "claude"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("GAMEDAY_CLAUDE_CLI", str(fake))
    assert session_mod.find_cli() == str(fake)


def test_missing_explicit_cli_path_is_not_silently_ignored(monkeypatch):
    monkeypatch.setenv("GAMEDAY_CLAUDE_CLI", "/nope/claude")
    assert session_mod.find_cli() is None


# --------------------------------------------------------------------------
# credential expiry — a present-but-dead token must not read as healthy
# --------------------------------------------------------------------------

def _write_creds(tmp_path, monkeypatch, expires_ms):
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / ".credentials.json").write_text(json.dumps(
        {"claudeAiOauth": {"accessToken": "x", "refreshToken": "y",
                           "expiresAt": expires_ms, "subscriptionType": "max"}}))
    monkeypatch.setattr(session_mod.Path, "home", staticmethod(lambda: home))
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_expired_cli_credential_is_reported_as_a_problem(tmp_path, monkeypatch):
    """The failure that actually bit: a credential file from June read as
    'available' right up until the first question 401'd."""
    import time
    _write_creds(tmp_path, monkeypatch, int((time.time() - 86_400) * 1000))
    state = session_mod.availability()
    assert state["credential"]["expired"] is True
    assert state["available"] is False
    assert any("claude setup-token" in p for p in state["problems"])


def test_live_cli_credential_is_not_flagged(tmp_path, monkeypatch):
    import time
    _write_creds(tmp_path, monkeypatch, int((time.time() + 86_400) * 1000))
    state = session_mod.availability()
    assert state["credential"]["expired"] is False
    assert not any("setup-token" in p for p in state["problems"])


def test_env_token_expiry_is_reported_as_unknown_not_assumed_good(monkeypatch):
    """An env-var token is opaque — say so rather than implying it was checked."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-opaque")
    assert session_mod.credential_expiry() == {"known": False}


def test_probe_reports_a_live_failure(manager, monkeypatch):
    async def get(session_id):
        sess = session_mod.Session(session_id=session_id, client=StubClient(script=[
            _result(is_error=True, api_error_status=401, result="expired")]))
        manager._sessions[session_id] = sess
        return sess

    monkeypatch.setattr(manager, "get", get)
    body = client.get("/api/chat/health?probe=1").json()
    assert body["probe"]["ok"] is False and body["available"] is False
    assert any("live probe failed" in p for p in body["problems"])


def test_probe_is_skipped_by_default(manager):
    assert "probe" not in client.get("/api/chat/health").json()

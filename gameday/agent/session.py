"""Conversation lifecycle for the draft agent.

One `ClaudeSDKClient` per browser session, so a follow-up ("what about him in
round 5?") lands in the same conversation. Each client owns a `claude`
subprocess, so sessions are capped and idle ones are evicted.

Auth resolution, in order:

1. `CLAUDE_CODE_OAUTH_TOKEN` — a long-lived token from `claude setup-token`.
   This is the one to use for the systemd service.
2. `ANTHROPIC_API_KEY` — pay-per-token. The swap to make if this app ever
   serves anyone other than its owner: subscription auth is for personal use,
   not for offering claude.ai rate limits to other people.
3. Whatever the CLI itself has in `~/.claude/.credentials.json`.

The tool surface is read-only by construction — `tools=["WebSearch",
"WebFetch"]` removes Bash/Read/Write/Edit from the model's context entirely,
rather than leaving them present but unapproved — which is what makes
`permission_mode="bypassPermissions"` safe here. `setting_sources=[]` keeps the
host's personal Claude settings, CLAUDE.md, and skills out of the conversation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from gameday.config import (AGENT_MAX_SESSIONS, AGENT_MAX_TURNS, AGENT_MODEL,
                            AGENT_SESSION_TTL_MIN)

log = logging.getLogger(__name__)

# Built-ins the agent keeps. Everything else is stripped from its context.
BUILTIN_TOOLS = ["WebSearch", "WebFetch"]


class AgentUnavailable(RuntimeError):
    """The agent can't run — SDK missing, CLI missing, or no credentials.

    Raised with a message meant for a human operator; the API surfaces it as a
    diagnosable state rather than a hung stream.
    """


def find_cli() -> str | None:
    """Absolute path to the `claude` binary.

    Resolved explicitly because the systemd user unit does not inherit a login
    shell's PATH, and on the Pi the CLI lives in `~/.local/bin`.
    """
    explicit = os.environ.get("GAMEDAY_CLAUDE_CLI")
    if explicit:
        return explicit if Path(explicit).exists() else None
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (Path.home() / ".local/bin/claude",
                      Path("/usr/local/bin/claude")):
        if candidate.exists():
            return str(candidate)
    return None


def auth_source() -> str | None:
    """Which credential the agent will use, by name (never the value)."""
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return "CLAUDE_CODE_OAUTH_TOKEN"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    if (Path.home() / ".claude" / ".credentials.json").exists():
        return "cli-credentials"
    return None


def credential_expiry() -> dict:
    """Expiry of the CLI credential file, when that's what we're falling back on.

    A credential file that exists but expired weeks ago reads as healthy to a
    presence check and then 401s on the first real question — which is exactly
    what happened on the Pi. This is the cheap half of the fix; a live probe
    (`/api/chat/health?probe=1`) is the expensive half.

    Only the CLI file can be checked offline. An env-var token is opaque, so
    `known` is False for it — absence of evidence, reported as such.
    """
    if auth_source() != "cli-credentials":
        return {"known": False}
    path = Path.home() / ".claude" / ".credentials.json"
    try:
        import json

        oauth = json.loads(path.read_text()).get("claudeAiOauth", {})
        expires_at = oauth.get("expiresAt")
        if not expires_at:
            return {"known": False}
        import datetime as dt

        when = dt.datetime.fromtimestamp(expires_at / 1000, dt.timezone.utc)
        expired = when < dt.datetime.now(dt.timezone.utc)
        return {"known": True, "expired": expired,
                "expires_at": when.isoformat(timespec="seconds"),
                "subscription": oauth.get("subscriptionType")}
    except (OSError, ValueError, KeyError, TypeError):
        return {"known": False}


def availability() -> dict:
    """Everything needed to diagnose a broken agent, without a live call."""
    try:
        import claude_agent_sdk  # noqa: F401
        sdk = True
    except ImportError:
        sdk = False
    cli = find_cli()
    auth = auth_source()
    expiry = credential_expiry()
    problems = []
    if not sdk:
        problems.append("claude-agent-sdk is not installed (pip install -e '.[agent]')")
    if not cli:
        problems.append("the `claude` CLI was not found on PATH or in ~/.local/bin")
    if not auth:
        problems.append("no credentials: set CLAUDE_CODE_OAUTH_TOKEN "
                        "(from `claude setup-token`) or ANTHROPIC_API_KEY")
    elif expiry.get("expired"):
        problems.append(
            f"the CLI credential expired at {expiry['expires_at']} — run "
            "`claude setup-token` and write CLAUDE_CODE_OAUTH_TOKEN to "
            "~/.config/gameday/agent.env")
    return {"available": not problems, "sdk_installed": sdk, "cli_path": cli,
            "auth_source": auth, "credential": expiry,
            "model": AGENT_MODEL, "problems": problems}


def build_options(cwd: str | None = None):
    """`ClaudeAgentOptions` for a draft-agent session."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions
    except ImportError as exc:
        raise AgentUnavailable(
            "claude-agent-sdk is not installed — `pip install -e '.[agent]'`"
        ) from exc

    from gameday.agent import tools
    from gameday.agent.prompt import SYSTEM_PROMPT

    cli = find_cli()
    if not cli:
        raise AgentUnavailable(
            "the `claude` CLI was not found. Install Claude Code, or set "
            "GAMEDAY_CLAUDE_CLI to its absolute path.")

    return ClaudeAgentOptions(
        model=AGENT_MODEL,
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={tools.SERVER_NAME: tools.build_server()},
        # Availability layer: only these built-ins exist for the model at all.
        tools=BUILTIN_TOOLS,
        # Permission layer: everything it can see, it may call without a prompt.
        allowed_tools=[*tools.TOOL_NAMES, *BUILTIN_TOOLS],
        permission_mode="bypassPermissions",
        # Don't inherit the host user's CLAUDE.md, settings, or skills — they
        # belong to unrelated projects and would leak into every answer.
        setting_sources=[],
        max_turns=AGENT_MAX_TURNS,
        cli_path=cli,
        cwd=cwd or str(Path.home()),
    )


@dataclass
class Session:
    """One live conversation."""

    session_id: str
    client: object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    turns: int = 0

    @property
    def idle_minutes(self) -> float:
        return (time.monotonic() - self.last_used) / 60.0


class SessionManager:
    """Keeps conversations alive between turns, and bounded in number."""

    def __init__(self, max_sessions: int = AGENT_MAX_SESSIONS,
                 ttl_minutes: float = AGENT_SESSION_TTL_MIN):
        self.max_sessions = max_sessions
        self.ttl_minutes = ttl_minutes
        self._sessions: dict[str, Session] = {}
        self._guard = asyncio.Lock()

    async def get(self, session_id: str) -> Session:
        """The session for this id, connecting a new one if needed."""
        async with self._guard:
            await self._evict_idle()
            existing = self._sessions.get(session_id)
            if existing is not None:
                existing.last_used = time.monotonic()
                return existing

            if len(self._sessions) >= self.max_sessions:
                # Drop the least recently used rather than refusing: this is a
                # single-user app, and a stale tab shouldn't block a live draft.
                oldest = min(self._sessions.values(), key=lambda s: s.last_used)
                log.info("evicting LRU agent session %s (idle %.1f min)",
                         oldest.session_id, oldest.idle_minutes)
                await self._close(oldest)

            from claude_agent_sdk import ClaudeSDKClient

            client = ClaudeSDKClient(options=build_options())
            await client.connect()
            session = Session(session_id=session_id, client=client)
            self._sessions[session_id] = session
            log.info("opened agent session %s (%d live)",
                     session_id, len(self._sessions))
            return session

    async def drop(self, session_id: str) -> bool:
        async with self._guard:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            await self._close(session)
            return True

    async def shutdown(self) -> None:
        async with self._guard:
            for session in list(self._sessions.values()):
                await self._close(session)

    async def _evict_idle(self) -> None:
        for session in list(self._sessions.values()):
            if session.idle_minutes > self.ttl_minutes:
                log.info("evicting idle agent session %s (%.1f min)",
                         session.session_id, session.idle_minutes)
                await self._close(session)

    async def _close(self, session: Session) -> None:
        self._sessions.pop(session.session_id, None)
        try:
            await session.client.disconnect()
        except Exception as exc:  # noqa: BLE001 — teardown must not raise
            log.warning("error closing agent session %s: %s",
                        session.session_id, exc)

    @property
    def live(self) -> int:
        return len(self._sessions)


MANAGER = SessionManager()

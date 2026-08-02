"""`/api/chat` — the draft agent, streamed to the dashboard over SSE.

Three event kinds go down the wire so the panel can show the agent working
rather than a spinner: `tool` (what it decided to look up), `text` (the answer),
and `done` (turn complete, with cost and turn count). A `status` event carries
failures that are worth reading rather than swallowing.

Everything here degrades: if `claude-agent-sdk` isn't installed or the token is
dead, the endpoint says so in a shape the panel can render. The dashboard must
keep working without the agent — the agent is additive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gameday.agent.session import MANAGER, AgentUnavailable, availability

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# A turn that hasn't produced anything in this long is stuck; fail it loudly
# rather than leaving the panel spinning through a draft pick.
TURN_TIMEOUT_S = 180.0

# Human-readable labels for the tool chips the panel renders.
_TOOL_LABELS = {
    "mcp__gameday__search_players": "looking up the player",
    "mcp__gameday__get_draft_board": "reading the draft board",
    "mcp__gameday__get_player": "pulling projection detail",
    "mcp__gameday__compare_players": "comparing players",
    "mcp__gameday__get_player_usage": "checking usage share",
    "mcp__gameday__get_model_track_record": "checking the model's track record",
    "mcp__gameday__get_player_replay": "checking past accuracy",
    "mcp__gameday__get_data_status": "checking data freshness",
    "WebSearch": "searching the news",
    "WebFetch": "reading an article",
}


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    # Where the user is in the SPA, so "what about him?" resolves without
    # making them retype a name.
    context: str | None = Field(default=None, max_length=500)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _tool_label(name: str) -> str:
    if name in _TOOL_LABELS:
        return _TOOL_LABELS[name]
    return f"using {name.rsplit('__', 1)[-1].replace('_', ' ')}"


def _framed(message: str, context: str | None) -> str:
    """The user's message, with the current view attached as context."""
    if not context:
        return message
    return (f"[The user is currently viewing: {context}]\n\n{message}")


async def _run_turn(req: ChatRequest) -> AsyncIterator[str]:
    """Drive one turn, translating SDK messages into SSE events."""
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

    try:
        session = await MANAGER.get(req.session_id)
    except AgentUnavailable as exc:
        yield _sse("status", {"level": "error", "message": str(exc),
                              "detail": availability()})
        return
    except Exception as exc:  # noqa: BLE001 — connect failures reach the user
        log.exception("agent session failed to start")
        yield _sse("status", {"level": "error",
                              "message": f"Could not start the agent: {exc}"})
        return

    # One turn at a time per conversation: the SDK client is not reentrant, and
    # a double-submit would interleave two answers.
    if session.lock.locked():
        yield _sse("status", {"level": "error",
                              "message": "That conversation is still answering."})
        return

    async with session.lock:
        try:
            await session.client.query(_framed(req.message, req.context))
            deadline = asyncio.get_running_loop().time() + TURN_TIMEOUT_S

            async for message in session.client.receive_response():
                if asyncio.get_running_loop().time() > deadline:
                    yield _sse("status", {"level": "error",
                                          "message": "The agent timed out."})
                    return

                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, ToolUseBlock):
                            yield _sse("tool", {"name": block.name,
                                                "label": _tool_label(block.name)})
                        elif isinstance(block, TextBlock) and block.text:
                            yield _sse("text", {"text": block.text})

                elif isinstance(message, ResultMessage):
                    session.turns += 1
                    yield _sse("done", {
                        "turns": message.num_turns,
                        "is_error": bool(message.is_error),
                        "cost_usd": message.total_cost_usd,
                        "stop_reason": message.stop_reason,
                        # A subscription-auth failure lands here (401) rather
                        # than as an exception, so surface it.
                        "api_error_status": message.api_error_status,
                        "result": message.result if message.is_error else None,
                    })
                    return

            yield _sse("done", {"turns": None, "is_error": False})

        except asyncio.CancelledError:  # client closed the tab mid-answer
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("agent turn failed")
            # A dead conversation shouldn't poison every later question.
            await MANAGER.drop(req.session_id)
            yield _sse("status", {"level": "error",
                                  "message": f"{type(exc).__name__}: {exc}"})


@router.post("")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Ask the draft agent a question; stream the answer back as SSE."""
    state = availability()
    if not state["available"]:
        # 503 with the diagnosis, not a hung stream.
        raise HTTPException(status_code=503, detail=state)

    return StreamingResponse(
        _run_turn(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


@router.delete("/{session_id}")
async def reset(session_id: str) -> dict:
    """Forget a conversation and free its subprocess."""
    dropped = await MANAGER.drop(session_id)
    return {"session_id": session_id, "dropped": dropped, "live": MANAGER.live}


@router.get("/health")
async def chat_health(probe: bool = False) -> dict:
    """Whether the agent can run, and if not, precisely what is missing.

    The default is a cheap offline check: SDK importable, CLI present, a
    credential configured (and, for the CLI credential file, not expired).

    `?probe=1` additionally asks the model one trivial question. That is the
    only way to prove an env-var token is live — it is opaque, so its validity
    can't be read off disk. Costs a few cents and a couple of seconds; use it
    after rotating a token, not on a dashboard poll.
    """
    state = {**availability(), "live_sessions": MANAGER.live}
    if not probe or not state["available"]:
        return state

    session_id = "__health_probe__"
    started = asyncio.get_running_loop().time()
    try:
        session = await MANAGER.get(session_id)
        await session.client.query("Reply with exactly: OK")
        from claude_agent_sdk import ResultMessage

        async for message in session.client.receive_response():
            if isinstance(message, ResultMessage):
                state["probe"] = {
                    "ok": not message.is_error,
                    "api_error_status": message.api_error_status,
                    "detail": message.result if message.is_error else None,
                    "seconds": round(
                        asyncio.get_running_loop().time() - started, 1),
                }
                break
    except Exception as exc:  # noqa: BLE001 — the probe reports, never raises
        state["probe"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        await MANAGER.drop(session_id)

    if not state["probe"].get("ok"):
        state["available"] = False
        state["problems"].append(
            f"live probe failed: {state['probe'].get('detail')}")
    return state

"""The agent's tool surface: Gameday's artifacts as an in-process MCP server.

Thin wrappers over `gameday.agent.data` — schema, description, and error
shaping only. The descriptions carry real weight: they are what the agent reads
when deciding whether a question needs the board, the usage history, the track
record, or a web search, so each one says *when* the tool is the right call and
not merely what it returns.

Every tool is read-only and marked as such, which lets the agent batch
independent lookups in one turn instead of serializing them.
"""

from __future__ import annotations

from typing import Annotated, Any

from claude_agent_sdk import ToolAnnotations, create_sdk_mcp_server, tool

from gameday.agent import data
from gameday.agent.data import ToolError

SERVER_NAME = "gameday"

_READ_ONLY = ToolAnnotations(readOnlyHint=True)


def _ok(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": data.as_text(payload)}]}


def _fail(message: str) -> dict[str, Any]:
    # is_error lets the agent distinguish "this lookup failed, try another
    # route" from "the answer is legitimately empty".
    return {"content": [{"type": "text", "text": message}], "is_error": True}


def _guard(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return _ok(fn(*args, **kwargs))
    except ToolError as exc:
        return _fail(str(exc))
    except Exception as exc:  # noqa: BLE001 — a tool bug must not kill the turn
        return _fail(f"{type(exc).__name__} while reading Gameday data: {exc}")


@tool(
    "search_players",
    "Find players by name (partial or full, punctuation and suffixes ignored). "
    "Returns player_id, position, team, and our board rank. Use this first for "
    "any question about a named player — the other tools take a player_id, and "
    "this is what disambiguates two players with similar names.",
    {"query": Annotated[str, "Full or partial player name, e.g. 'Nabers'"]},
    annotations=_READ_ONLY,
)
async def search_players(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.search, args["query"])


@tool(
    "get_draft_board",
    "The VORP-ranked draft board with each player's season projection, "
    "floor-ceiling envelope, ADP, value vs the market, and projection spread. "
    "Use for ranking, tier, and 'who should I target' questions. Optional args: "
    "'position' (QB/RB/WR/TE, default ALL), 'tier' ('sleeper' or 'reach' to "
    "show only players we disagree with the market about), 'limit' (default 40).",
    {},
    annotations=_READ_ONLY,
)
async def get_draft_board(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.board,
                  position=args.get("position", "ALL") or "ALL",
                  tier=args.get("tier", "") or "",
                  limit=int(args.get("limit", 40) or 40))


@tool(
    "get_player",
    "Everything the board knows about one player: season projection (sum of "
    "weekly medians), the p25-p75 envelope, weekly projections, bye week, VORP, "
    "our rank, the market's rank and ADP, and each outside source's projection. "
    "Takes a player_id from search_players, or a name if it is unambiguous.",
    {"player": Annotated[str, "A player_id from search_players, or a name"]},
    annotations=_READ_ONLY,
)
async def get_player(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.player, args["player"])


@tool(
    "compare_players",
    "Two to four players side by side on projection, envelope, VORP, rank, ADP, "
    "and value vs market. Use for 'who do I take' and 'is X or Y better here' — "
    "the numbers are computed across the whole board, so cross-position "
    "comparisons are valid. Pass 'players' as a list of ids or names.",
    {"players": Annotated[list[str], "2-4 player_ids or names"]},
    annotations=_READ_ONLY,
)
async def compare_players(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.compare, list(args["players"]))


@tool(
    "get_player_usage",
    "A player's recent snap share, target share, carry share, air-yards share, "
    "and WOPR, week by week. This is the volume a projection is built on — use "
    "it to explain WHY someone is ranked where they are, or to check whether a "
    "projection rests on usage that has since changed.",
    {"player": Annotated[str, "A player_id from search_players, or a name"]},
    annotations=_READ_ONLY,
)
async def get_player_usage(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.usage, args["player"])


@tool(
    "get_model_track_record",
    "How this model actually scored on past seasons: MAE, skill vs a "
    "trailing-average baseline, p10-p90 coverage, broken out by season phase, "
    "player experience, and whether the player changed teams. Use it before "
    "making a confidence claim — it is the difference between 'the model likes "
    "him' and 'the model has been measurably good at this kind of call'. "
    "Optional 'season' (int) narrows to one year.",
    {},
    annotations=_READ_ONLY,
)
async def get_model_track_record(args: dict[str, Any]) -> dict[str, Any]:
    season = args.get("season")
    return _guard(data.track_record, int(season) if season else None)


@tool(
    "get_player_replay",
    "One player's week-by-week projection vs what they actually scored in a "
    "past season, plus how often the actual landed inside the p25-p75 band. "
    "Use when the question is whether the model has historically been right "
    "about THIS player, rather than about players in general.",
    {"player": Annotated[str, "A player_id from search_players, or a name"],
     "season": Annotated[int, "Season year, e.g. 2024"]},
    annotations=_READ_ONLY,
)
async def get_player_replay(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.player_history, args["player"], int(args["season"]))


@tool(
    "get_data_status",
    "When the projections, the market data, and the model were last updated, "
    "and whether the last refresh ran or self-gated. Call this when the answer "
    "depends on how current the numbers are — especially before asserting "
    "anything about a player's current role or team.",
    {},
    annotations=_READ_ONLY,
)
async def get_data_status(args: dict[str, Any]) -> dict[str, Any]:
    return _guard(data.data_status)


TOOLS = [
    search_players,
    get_draft_board,
    get_player,
    compare_players,
    get_player_usage,
    get_model_track_record,
    get_player_replay,
    get_data_status,
]

# Fully-qualified names, as the SDK exposes them to the model.
TOOL_NAMES = [f"mcp__{SERVER_NAME}__{t.name}" for t in TOOLS]


def build_server():
    """The in-process MCP server the session hands to ClaudeAgentOptions."""
    return create_sdk_mcp_server(name=SERVER_NAME, version="1.0.0", tools=TOOLS)

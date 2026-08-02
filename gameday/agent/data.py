"""Data accessors behind the agent's tools — plain functions, no SDK.

Every accessor delegates to the loaders in `gameday.api.server`, which already
own artifact discovery, mtime caching, and the demo-fixture fallback. Nothing
here re-reads parquet or re-derives VORP; a number the agent quotes is the same
number the dashboard renders, by construction.

Keeping this module SDK-free is deliberate: the tool surface is the part most
likely to be wrong, and it stays unit-testable without `claude_agent_sdk`
installed and without a model in the loop.

`gameday.api.server` is imported lazily inside `_srv()` because the API mounts
the chat router, which reaches back here — importing at module scope would make
that cycle load-bearing.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from gameday.config import FORECASTS_DIR
from gameday.data.market import merge_name

# A player search that returns half the league isn't an answer; cap it so the
# agent gets a short list it can disambiguate against.
SEARCH_LIMIT = 12
MAX_COMPARE = 4


class ToolError(Exception):
    """A failure the agent should read and act on, not a crash.

    Raised with a message written for the model — what went wrong and, where
    there is one, the command that fixes it.
    """


def _srv():
    """The API module, imported on first use (see the cycle note above)."""
    from gameday.api import server

    return server


def _board() -> list[dict]:
    """The full VORP-ranked board with market columns attached.

    This is `GET /api/draft` with no filters — same ranks, same value figures,
    same tier flags. Cached against the artifacts' mtimes so repeated tool calls
    within one answer don't rebuild it.
    """
    srv = _srv()
    paths = [FORECASTS_DIR / "latest_season.parquet",
             FORECASTS_DIR / "latest_market.parquet"]
    try:
        return srv._mtime_cached(
            "agent:board", paths,
            lambda: srv.draft_board(position="ALL", limit=10_000)["players"])
    except Exception as exc:  # HTTPException when the season artifact is absent
        raise ToolError(
            "No draft board available yet. The season projection artifact is "
            "missing — it is written by `gameday refresh` (in the offseason, "
            "`gameday refresh --horizon-days 90`)."
        ) from exc


def _board_meta() -> dict:
    srv = _srv()
    data = srv.draft_board(position="ALL", limit=1)
    return {k: data[k] for k in
            ("season", "first_week", "last_week", "replacement",
             "model_version", "generated_at", "market")}


def _index() -> dict[str, dict]:
    return {p["player_id"]: p for p in _board()}


def _with_position_rank(entry: dict) -> dict:
    """Add `position_rank` (TE4, RB11) alongside the overall rank.

    Without it the agent infers a positional rank from the overall one and gets
    it wrong — an observed failure, not a hypothetical: Kyle Pitts at overall
    rank 19 was reported as "TE19" when he is TE4.
    """
    same_position = [p for p in _board() if p["position"] == entry["position"]]
    same_position.sort(key=lambda p: -p["vorp"])
    rank = next((i for i, p in enumerate(same_position, start=1)
                 if p["player_id"] == entry["player_id"]), None)
    return {**entry, "position_rank": rank,
            "position_rank_label": f"{entry['position']}{rank}" if rank else None}


def resolve(who: str) -> dict:
    """A player id or a name -> that player's board entry.

    Accepts an id directly so the agent can chain calls without a second
    lookup, and falls back to name matching so it can act on what a user typed.
    Ambiguity is an error carrying the candidates rather than a silent pick —
    guessing between two similar names is how a draft-day answer goes wrong.
    """
    index = _index()
    if who in index:
        return index[who]
    matches = search(who)
    if not matches:
        raise ToolError(
            f"No player matching {who!r} is on the draft board. Try "
            "search_players with a partial last name.")
    if len(matches) > 1 and matches[0]["match"] != "exact":
        names = ", ".join(f"{m['name']} ({m['position']} {m['team']})" for m in matches[:5])
        raise ToolError(f"{who!r} is ambiguous — could be: {names}. "
                        "Call get_player with the player_id you mean.")
    return index[matches[0]["player_id"]]


def search(query: str) -> list[dict]:
    """Players whose name matches `query`, best match first.

    Matches on the punctuation- and suffix-stripped form used across the market
    join, so "AJ Brown", "A.J. Brown", and "Marvin Harrison Jr" all land.
    """
    want = merge_name(query)
    if not want:
        return []
    # merge_name strips spaces as well as punctuation and suffixes, so a
    # substring test covers first name, surname, and any contiguous fragment:
    # "brown" and "chase" both hit "chasebrown".
    out = []
    for p in _board():
        name = merge_name(p["name"])
        if name == want:
            kind = "exact"
        elif want in name:
            kind = "partial"
        else:
            continue
        out.append({"player_id": p["player_id"], "name": p["name"],
                    "position": p["position"], "team": p["team"],
                    "our_rank": p["our_rank"], "match": kind})
    # Exact first, then by our own board rank so the most draftable
    # candidate leads an ambiguous list.
    out.sort(key=lambda m: (m["match"] != "exact",
                            m["our_rank"] is None, m["our_rank"] or 1e9))
    return out[:SEARCH_LIMIT]


def board(position: str = "ALL", tier: str = "", limit: int = 40) -> dict:
    """A slice of the draft board — the same payload `/api/draft` serves."""
    srv = _srv()
    try:
        data = srv.draft_board(position=position.upper() or "ALL",
                               tier=tier, limit=max(1, min(limit, 300)))
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        raise ToolError(f"Could not build the draft board: {detail}") from exc
    return data


def player(who: str) -> dict:
    """One player's full board entry: projection, envelope, VORP, and market."""
    return _with_position_rank(resolve(who))


def compare(whos: list[str]) -> dict:
    """Several players side by side on the fields that decide a pick."""
    if len(whos) < 2:
        raise ToolError("compare_players needs at least two players.")
    if len(whos) > MAX_COMPARE:
        raise ToolError(f"compare_players takes at most {MAX_COMPARE} players.")
    picked = [_with_position_rank(resolve(w)) for w in whos]
    fields = ["player_id", "name", "position", "team", "bye", "games",
              "total_p50", "total_floor", "total_ceiling", "vorp",
              "our_rank", "position_rank_label", "market_rank", "value", "value_tier",
              "espn_adp", "espn_proj_pts", "fft_proj_ppr", "sleeper_rank",
              "proj_spread", "proj_sources"]
    return {"players": [{f: p.get(f) for f in fields} for p in picked],
            "note": "Ranks and value are computed across the whole board, so "
                    "they are comparable across positions."}


def usage(who: str) -> dict:
    """Snap / target / carry share history — the volume behind a projection."""
    entry = resolve(who)
    srv = _srv()
    try:
        rows = srv.player_usage(entry["player_id"])
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        raise ToolError(
            f"No usage rows for {entry['name']}: {detail}") from exc
    return {"player_id": entry["player_id"], "name": entry["name"],
            "position": entry["position"], "team": entry["team"],
            "weeks": rows["weeks"],
            "note": "Shares are of the player's own team. Rows are the most "
                    "recent played weeks, oldest first."}


def track_record(season: int | None = None) -> dict:
    """How the model actually scored on past seasons, overall and by segment.

    Exists so a claim about reliability can be checked instead of asserted.
    `skill_vs_naive` is the fraction better than a last-8-games average; a
    negative value means the model was worse than that baseline for the segment.
    """
    srv = _srv()
    seasons = [s["season"] for s in srv.replay_seasons()["seasons"]]
    if not seasons:
        raise ToolError(
            "No replay artifacts yet — generate them with "
            "`gameday replay --season <year> --compare`.")
    wanted = [season] if season else seasons
    out = {}
    for yr in wanted:
        try:
            card = srv.replay_scorecard(yr)
        except Exception:
            continue
        out[str(yr)] = card.get("adjusted", card)
    if not out:
        raise ToolError(f"No replay scorecard for {season}. Have: {seasons}.")
    return {"seasons_available": seasons, "scorecards": out,
            "note": "skill_vs_naive is the improvement over a trailing-average "
                    "baseline; coverage80 is the share of actuals that landed "
                    "inside the p10-p90 band (0.80 is calibrated)."}


def player_history(who: str, season: int) -> dict:
    """One player's week-by-week forecast-vs-actual for a replayed season."""
    entry = resolve(who)
    srv = _srv()
    try:
        df = srv._load_replay(season)
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        raise ToolError(f"No replay for {season}: {detail}") from exc
    rows = df[df["player_id"] == entry["player_id"]].sort_values("week")
    if rows.empty:
        raise ToolError(
            f"{entry['name']} has no {season} replay rows (they may not have "
            "been on a roster that season).")
    weeks = [{"week": int(r["week"]), "opponent": r.get("opponent_team"),
              "projected_p50": round(float(r["fantasy_points_p50"]), 1),
              "band_p25_p75": [round(float(r["fantasy_points_p25"]), 1),
                               round(float(r["fantasy_points_p75"]), 1)],
              "actual": None if pd.isna(r["fantasy_points"]) else round(float(r["fantasy_points"]), 1)}
             for _, r in rows.iterrows()]
    hits = [w for w in weeks if w["actual"] is not None
            and w["band_p25_p75"][0] <= w["actual"] <= w["band_p25_p75"][1]]
    scored = [w for w in weeks if w["actual"] is not None]
    return {
        "player_id": entry["player_id"], "name": entry["name"], "season": season,
        "weeks": weeks,
        "summary": {
            "weeks_scored": len(scored),
            "in_p25_p75": len(hits),
            "mean_error": round(
                sum(w["actual"] - w["projected_p50"] for w in scored) / len(scored), 2)
            if scored else None,
        },
        "note": "mean_error is actual minus projected: positive means the model "
                "was low on this player.",
    }


def data_status() -> dict:
    """How fresh the numbers are — so staleness can be stated, not implied."""
    srv = _srv()
    health = srv.health()
    status_path = FORECASTS_DIR / "refresh_status.json"
    try:
        meta = _board_meta()
    except Exception:
        meta = {}
    market = meta.get("market") or {}
    return {
        "model": health.get("model", {}).get("version") if health.get("model") else None,
        "forecasts": health.get("forecasts"),
        "last_refresh": health.get("refresh"),
        "board": {k: meta.get(k) for k in ("season", "first_week", "last_week")},
        "market": {"available": market.get("available", False),
                   "fetched_at": market.get("fetched_at"),
                   "coverage": market.get("coverage"),
                   "ranked_players": market.get("ranked_players")},
        "refresh_status_path_exists": status_path.exists(),
        "note": "A skipped_reason on last_refresh means the nightly run "
                "self-gated (no game within the horizon) — normal in the "
                "offseason, and it means the projections have not moved since.",
    }


def as_text(payload: Any) -> str:
    """Tool payload -> the string the model reads. JSON keeps numbers exact."""
    return json.dumps(payload, indent=2, default=str)

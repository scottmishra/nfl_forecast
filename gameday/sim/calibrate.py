"""Team simulation profiles calibrated from historical player-weeks.

Each team gets a TeamProfile: pace, play-calling lean, offensive efficiency,
defensive adjustments, and a personnel table (usage shares per player) — all
estimated from the team's trailing games and shrunk toward league means so
thin data never produces a degenerate simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRAILING_GAMES = 8

# League anchors (shrinkage targets). Roughly 2020s-NFL-shaped.
LEAGUE = dict(
    plays_per_game=63.0,
    pass_rate=0.575,
    comp_rate=0.645,
    yards_per_comp=11.2,
    yards_per_carry=4.3,
    int_per_att=0.022,
    sack_rate=0.065,
    points_per_game=22.0,
)

# How many of each position the sim keeps on the active roster.
ROSTER_CAP = {"QB": 1, "RB": 2, "WR": 3, "TE": 2}


@dataclass
class SimPlayer:
    player_id: str
    name: str
    position: str
    carry_share: float      # share of team carries
    target_share: float     # share of team targets
    catch_rate: float
    yards_per_carry: float
    yards_per_catch: float
    snap_weight: float      # probability of being on the field for a snap


@dataclass
class TeamProfile:
    team: str
    plays_per_game: float
    pass_rate: float        # neutral-situation pass rate
    comp_rate: float
    yards_per_comp: float
    yards_per_carry: float
    int_per_att: float
    def_pass_adj: float = 1.0   # multiplier on opponent passing efficiency
    def_rush_adj: float = 1.0
    def_points_allowed: float = LEAGUE["points_per_game"]
    players: list[SimPlayer] = field(default_factory=list)


def _shrink(value: float, anchor: float, n: int, k: float = 6.0) -> float:
    """Bayesian-ish shrinkage: k pseudo-games toward the league anchor."""
    if not np.isfinite(value):
        return anchor
    w = n / (n + k)
    return w * value + (1 - w) * anchor


def _recent(team_df: pd.DataFrame) -> pd.DataFrame:
    weeks = (team_df[["season", "week"]].drop_duplicates()
             .sort_values(["season", "week"]).tail(TRAILING_GAMES))
    return team_df.merge(weeks, on=["season", "week"])


def build_profiles(player_weeks: pd.DataFrame, games: pd.DataFrame) -> dict[str, TeamProfile]:
    """Calibrate a TeamProfile for every team present in the data."""
    profiles: dict[str, TeamProfile] = {}

    # Defensive adjustments from points allowed in played games.
    played = games[games["home_score"].notna()]
    pa = pd.concat([
        played[["home_team", "away_score"]].rename(columns={"home_team": "team", "away_score": "pts"}),
        played[["away_team", "home_score"]].rename(columns={"away_team": "team", "home_score": "pts"}),
    ])
    pts_allowed = pa.groupby("team")["pts"].apply(
        lambda s: s.tail(TRAILING_GAMES).mean())

    # Defense-vs-pass/rush: fantasy points conceded to QBs vs RBs (usage proxy).
    def _def_adj(pos_list: list[str]) -> pd.Series:
        conceded = (player_weeks[player_weeks["position"].isin(pos_list)]
                    .groupby(["opponent_team", "season", "week"])["fantasy_points"].sum()
                    .groupby("opponent_team").apply(lambda s: s.tail(TRAILING_GAMES).mean()))
        return conceded / conceded.mean()

    pass_adj = _def_adj(["QB", "WR", "TE"])
    rush_adj = _def_adj(["RB"])

    for team, tdf in player_weeks.groupby("team"):
        recent = _recent(tdf)
        n_games = recent[["season", "week"]].drop_duplicates().shape[0]
        per_game = recent.groupby(["season", "week"])

        attempts = per_game["attempts"].sum().mean()
        carries = per_game["carries"].sum().mean()
        completions = per_game["completions"].sum().mean()
        pass_yards = per_game["passing_yards"].sum().mean()
        rush_yards = per_game["rushing_yards"].sum().mean()
        ints = per_game["interceptions"].sum().mean()

        plays = _shrink(attempts + carries + 6.0, LEAGUE["plays_per_game"], n_games)
        pass_rate = _shrink(attempts / max(attempts + carries, 1),
                            LEAGUE["pass_rate"], n_games)
        comp_rate = _shrink(completions / max(attempts, 1), LEAGUE["comp_rate"], n_games)
        ypcomp = _shrink(pass_yards / max(completions, 1), LEAGUE["yards_per_comp"], n_games)
        ypc = _shrink(rush_yards / max(carries, 1), LEAGUE["yards_per_carry"], n_games)
        int_rate = _shrink(ints / max(attempts, 1), LEAGUE["int_per_att"], n_games)

        profiles[team] = TeamProfile(
            team=team, plays_per_game=plays, pass_rate=pass_rate, comp_rate=comp_rate,
            yards_per_comp=ypcomp, yards_per_carry=ypc, int_per_att=int_rate,
            def_pass_adj=float(np.clip(pass_adj.get(team, 1.0), 0.75, 1.25)),
            def_rush_adj=float(np.clip(rush_adj.get(team, 1.0), 0.75, 1.25)),
            def_points_allowed=float(pts_allowed.get(team, LEAGUE["points_per_game"])),
            players=_personnel(recent),
        )
    return profiles


def _personnel(recent: pd.DataFrame) -> list[SimPlayer]:
    """Active roster with usage shares, capped per ROSTER_CAP by recent usage."""
    agg = recent.groupby(["player_id", "player_display_name", "position"]).agg(
        carries=("carries", "sum"), targets=("targets", "sum"),
        receptions=("receptions", "sum"), rec_yards=("receiving_yards", "sum"),
        rush_yards=("rushing_yards", "sum"), games=("week", "count"),
    ).reset_index()

    team_carries = max(agg["carries"].sum(), 1)
    team_targets = max(agg["targets"].sum(), 1)

    players: list[SimPlayer] = []
    for pos, cap in ROSTER_CAP.items():
        pos_df = agg[agg["position"] == pos].copy()
        pos_df["usage"] = pos_df["carries"] + pos_df["targets"]
        if pos == "QB":  # starters throw; rank by games played then usage
            pos_df = pos_df.sort_values(["games", "usage"], ascending=False)
        else:
            pos_df = pos_df.sort_values("usage", ascending=False)
        for rank, (_, p) in enumerate(pos_df.head(cap).iterrows()):
            n = max(int(p.games), 1)
            catch = _shrink(p.receptions / max(p.targets, 1), 0.66, n)
            ypcatch = _shrink(p.rec_yards / max(p.receptions, 1),
                              LEAGUE["yards_per_comp"], n)
            ypcarry = _shrink(p.rush_yards / max(p.carries, 1),
                              LEAGUE["yards_per_carry"], n)
            snap = {"QB": 1.0, "RB": (0.68, 0.38), "WR": (0.92, 0.85, 0.62),
                    "TE": (0.82, 0.38)}[pos]
            snap_weight = snap if isinstance(snap, float) else snap[min(rank, len(snap) - 1)]
            players.append(SimPlayer(
                player_id=p.player_id, name=p.player_display_name, position=pos,
                carry_share=p.carries / team_carries,
                target_share=p.targets / team_targets,
                catch_rate=float(np.clip(catch, 0.4, 0.85)),
                yards_per_carry=float(np.clip(ypcarry, 2.5, 6.5)),
                yards_per_catch=float(np.clip(ypcatch, 5.0, 18.0)),
                snap_weight=float(snap_weight),
            ))

    # Renormalize shares over the kept roster so sampling weights sum to 1.
    tot_carry = sum(p.carry_share for p in players) or 1.0
    tot_target = sum(p.target_share for p in players) or 1.0
    for p in players:
        p.carry_share /= tot_carry
        p.target_share /= tot_target
    return players

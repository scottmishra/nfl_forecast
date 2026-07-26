"""Synthetic league generator for offline demos and tests.

Produces player-week stat lines and schedules in the exact shape of the
nflverse loaders, driven by latent player skill, team strength, opponent
defense, venue, and weather — so the feature pipeline and models have real
signal to find. Uses the real 32 teams but clearly fictional players.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gameday.data.teams import TEAMS

FIRST = ["Dash", "Bo", "Zeke", "Trey", "Marcus", "Jalen", "Knox", "Cade", "Rico", "Ty",
         "Devon", "Amari", "Jett", "Cruz", "Maverick", "King", "Ace", "Duke", "Nico", "Blaze"]
LAST = ["Blackwood", "Steele", "Fontaine", "Maddox", "Rivers", "Stone", "Vance", "Cole",
        "Hollis", "Draper", "Kincaid", "Mercer", "Quill", "Ashford", "Bright", "Calloway",
        "Delacroix", "Everhart", "Falco", "Granger"]

ROSTER_SHAPE = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

# Per-position weekly stat means for an average starter, scaled by skill/context.
BASELINES = {
    "QB": dict(attempts=33, completions=21, passing_yards=235, passing_tds=1.5,
               interceptions=0.8, carries=4, rushing_yards=18, rushing_tds=0.15),
    "RB": dict(carries=13, rushing_yards=55, rushing_tds=0.45, targets=3.5,
               receptions=2.6, receiving_yards=20, receiving_tds=0.08),
    "WR": dict(targets=6.5, receptions=4.2, receiving_yards=52, receiving_tds=0.35),
    "TE": dict(targets=4.5, receptions=3.2, receiving_yards=34, receiving_tds=0.25),
}


def _fantasy_ppr(row: pd.Series) -> float:
    return round(
        row.get("passing_yards", 0) * 0.04 + row.get("passing_tds", 0) * 4
        - row.get("interceptions", 0) * 2 + row.get("rushing_yards", 0) * 0.1
        + row.get("rushing_tds", 0) * 6 + row.get("receptions", 0) * 1.0
        + row.get("receiving_yards", 0) * 0.1 + row.get("receiving_tds", 0) * 6,
        2,
    )


def generate(seasons: list[int] | None = None, weeks: int = 17, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (player_weeks, games). Last season's final week has no scores —
    that's the upcoming slate the app forecasts."""
    rng = np.random.default_rng(seed)
    seasons = seasons or [2023, 2024, 2025]
    teams = list(TEAMS)

    # Latent strengths, persistent across seasons with mild drift.
    off = {t: rng.normal(0, 0.18) for t in teams}
    deff = {t: rng.normal(0, 0.18) for t in teams}   # + = tough defense
    rosters, pid = [], 0
    used_names: set[str] = set()
    for t in teams:
        for pos, n in ROSTER_SHAPE.items():
            for slot in range(n):
                while True:
                    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                    if name not in used_names:
                        used_names.add(name)
                        break
                pid += 1
                # slot 0 is the workhorse; depth pieces get smaller roles
                role = 1.0 if slot == 0 else rng.uniform(0.45, 0.75)
                rosters.append(dict(
                    player_id=f"DEMO{pid:04d}", player_display_name=name, position=pos,
                    team=t, skill=rng.normal(0, 0.22), role=role,
                ))
    roster_df = pd.DataFrame(rosters)

    game_rows, stat_rows = [], []
    for season in seasons:
        for t in teams:  # season-over-season drift
            off[t] += rng.normal(0, 0.05)
            deff[t] += rng.normal(0, 0.05)
        for week in range(1, weeks + 1):
            order = rng.permutation(teams)
            upcoming = season == seasons[-1] and week == weeks
            for i in range(0, 32, 2):
                home, away = order[i], order[i + 1]
                venue = TEAMS[home]
                dome = venue["roof"] in ("dome", "retractable")
                temp = 21.0 if dome else float(rng.normal(12, 9))
                wind = 0.0 if dome else float(np.clip(rng.gamma(2.2, 4.5), 0, 45))
                month = 9 + (week - 1) // 4
                day = 1 + ((week - 1) % 4) * 7
                year = season if month <= 12 else season + 1  # January spillover
                month = month if month <= 12 else month - 12
                game = dict(
                    game_id=f"{season}_{week:02d}_{away}_{home}", season=season, week=week,
                    gameday=f"{year}-{month:02d}-{day:02d}", gametime="13:00",
                    home_team=home, away_team=away,
                    home_rest=int(rng.choice([6, 7, 7, 7, 10, 13])),
                    away_rest=int(rng.choice([6, 7, 7, 7, 10, 13])),
                    roof=venue["roof"], temp=round(temp, 1), wind=round(wind, 1),
                )
                if upcoming:
                    game["home_score"] = game["away_score"] = np.nan
                    game_rows.append(game)
                    continue

                scores = {}
                for team, opp, is_home in ((home, away, 1), (away, home, 0)):
                    # Team environment multiplier: offense vs defense, home edge,
                    # weather drag on passing efficiency.
                    env = np.exp(off[team] - deff[opp] + 0.045 * is_home)
                    weather_drag = 1.0 - (0.004 * max(wind - 15, 0) + 0.003 * max(-temp, 0))
                    team_pts = 0.0
                    for _, p in roster_df[roster_df.team == team].iterrows():
                        stats = {}
                        base = BASELINES[p.position]
                        pmult = np.exp(p.skill) * p.role * env
                        for stat, mu in base.items():
                            m = mu * pmult
                            if stat in ("passing_yards", "receiving_yards") or p.position == "QB" and stat == "attempts":
                                m *= weather_drag
                            if stat.endswith("_tds") or stat == "interceptions":
                                stats[stat] = int(rng.poisson(m))
                            else:
                                stats[stat] = max(0, round(float(rng.normal(m, mu * 0.45)), 0))
                        # keep counting stats coherent: completions are a
                        # binomial draw so completion % stays league-realistic
                        if "completions" in stats and "attempts" in stats:
                            comp_p = float(np.clip(0.64 * np.exp(p.skill * 0.3), 0.5, 0.75))
                            stats["completions"] = int(rng.binomial(int(stats["attempts"]), comp_p))
                        if "receptions" in stats:
                            stats["receptions"] = min(stats["receptions"], stats.get("targets", stats["receptions"]))
                        row = dict(player_id=p.player_id, player_display_name=p.player_display_name,
                                   position=p.position, team=team, season=season, week=week,
                                   opponent_team=opp, **stats)
                        row["fantasy_points"] = _fantasy_ppr(pd.Series(row))
                        stat_rows.append(row)
                        team_pts += 6.2 * (stats.get("passing_tds", 0) + stats.get("rushing_tds", 0) + stats.get("receiving_tds", 0)) / 2
                    scores[team] = int(np.clip(rng.normal(10 + team_pts, 4), 0, 62))
                game["home_score"], game["away_score"] = scores[home], scores[away]
                game_rows.append(game)

    player_weeks = pd.DataFrame(stat_rows)
    for col in ["completions", "attempts", "passing_yards", "passing_tds", "interceptions",
                "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
                "receiving_yards", "receiving_tds"]:
        if col not in player_weeks.columns:
            player_weeks[col] = 0.0
        player_weeks[col] = player_weeks[col].fillna(0.0)
    return player_weeks, pd.DataFrame(game_rows)

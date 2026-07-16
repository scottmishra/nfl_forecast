"""Discrete-event play-by-play game simulator.

Each event is one play. Game state carries (quarter, clock, down, distance,
field position, score, possession); a situational play-calling policy picks
pass/run/kick, calibrated distributions resolve the outcome, and a clock
model advances time. One call = one replicate; run many via sim/run.py.

Deliberate simplifications (documented, not hidden): no penalties, no
2-point tries, no onside kicks, single sudden-score overtime. These move
totals by a point or two; the structural questions this answers (play mix,
snap counts, usage, win probability) are robust to them.

Field position convention: `yardline` is distance from the offense's own
goal line (0) to the opponent's (100). yardline >= 100 on a gained play = TD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gameday.sim.calibrate import LEAGUE, SimPlayer, TeamProfile

QUARTER_SEC = 900
KICKOFF_YARDLINE = 25
PUNT_NET = 40.0

# League situational pass-rate curve; team lean shifts it in logit space.
BASE_PASS_BY_DOWN = {1: 0.50, 2: 0.55, 3: 0.72, 4: 0.80}


@dataclass
class PlayerBox:
    """Accumulated box score for one player in one replicate."""
    snaps: int = 0
    carries: int = 0
    rush_yards: float = 0.0
    rush_tds: int = 0
    targets: int = 0
    receptions: int = 0
    rec_yards: float = 0.0
    rec_tds: int = 0
    pass_attempts: int = 0
    pass_yards: float = 0.0
    pass_tds: int = 0
    interceptions: int = 0


@dataclass
class TeamResult:
    points: int = 0
    plays: int = 0
    pass_plays: int = 0
    run_plays: int = 0
    pass_by_down: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    plays_by_down: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    drives: int = 0
    drive_outcomes: dict = field(default_factory=lambda: dict(
        td=0, fg=0, punt=0, turnover=0, downs=0, end_half=0))
    box: dict[str, PlayerBox] = field(default_factory=dict)


@dataclass
class SimResult:
    home: TeamResult
    away: TeamResult


class _Sim:
    HOME_YARDS_EDGE = 1.04   # modest home-field bump on efficiency
    HOME_COMP_EDGE = 0.015

    def __init__(self, home: TeamProfile, away: TeamProfile, rng: np.random.Generator):
        self.rng = rng
        self.profiles = {"home": home, "away": away}
        self.results = {"home": TeamResult(), "away": TeamResult()}
        for side in ("home", "away"):
            for p in self.profiles[side].players:
                self.results[side].box[p.player_id] = PlayerBox()
        # Pace: derive per-play clock runoff so combined snaps land near the
        # two teams' calibrated plays/game (minus ~450s of kick/punt overhead).
        # The 0.87 divisor compensates for incompletions burning less clock,
        # so realized snap counts land on target rather than ~10% over.
        target_snaps = home.plays_per_game + away.plays_per_game
        self.base_runoff = max((3600.0 - 450.0) / max(target_snaps, 80) / 0.87, 18.0)

    # ---------------- policies ----------------

    def _pass_prob(self, side: str, down: int, ydstogo: float, diff: int, sec_left: int) -> float:
        base = BASE_PASS_BY_DOWN[down]
        logit = np.log(base / (1 - base))
        team = self.profiles[side].pass_rate
        logit += np.log(team / (1 - team)) - np.log(LEAGUE["pass_rate"] / (1 - LEAGUE["pass_rate"]))
        if down >= 2 and ydstogo >= 8:
            logit += 0.7
        elif ydstogo <= 2:
            logit -= 0.9
        if sec_left < 480 and diff < -4:   # trailing late: throw
            logit += 0.9
        if sec_left < 480 and diff > 7:    # protecting a lead: run clock
            logit -= 0.8
        return float(1 / (1 + np.exp(-logit)))

    def _fourth_down(self, yardline: float, ydstogo: float, diff: int, sec_left: int) -> str:
        kick_dist = (100 - yardline) + 17
        desperate = diff < 0 and sec_left < 300
        if desperate and (kick_dist > 55 or diff < -3):
            return "go"
        if kick_dist <= 55:
            return "fg"
        if ydstogo <= 2 and yardline >= 45 and self.rng.random() < 0.55:
            return "go"
        return "punt"

    # ---------------- personnel ----------------

    def _snap_players(self, side: str) -> list[SimPlayer]:
        """Sample the offensive skill players on the field for one snap."""
        on_field = []
        for p in self.profiles[side].players:
            if p.position == "QB" or self.rng.random() < p.snap_weight:
                on_field.append(p)
        return on_field

    def _weighted_pick(self, players: list[SimPlayer], attr: str) -> SimPlayer | None:
        weights = np.array([getattr(p, attr) for p in players])
        if weights.sum() <= 0:
            return None
        return players[self.rng.choice(len(players), p=weights / weights.sum())]

    # ---------------- play resolution ----------------

    def _resolve_pass(self, side: str, dside: str) -> tuple[float, bool, bool, SimPlayer | None]:
        """Returns (yards, turnover, incomplete, receiver) and updates boxes."""
        off, deff = self.profiles[side], self.profiles[dside]
        res = self.results[side]
        qb = next(p for p in off.players if p.position == "QB")
        box = res.box[qb.player_id]

        if self.rng.random() < LEAGUE["sack_rate"]:
            return float(-abs(self.rng.normal(6.5, 2.0))), False, False, None

        box.pass_attempts += 1
        # def_pass_adj > 1 = generous defense: fewer INTs, easier completions
        if self.rng.random() < off.int_per_att * (2.0 - deff.def_pass_adj) ** 0.5:
            box.interceptions += 1
            return 0.0, True, False, None

        target = self._weighted_pick(
            [p for p in off.players if p.position != "QB"], "target_share")
        if target is None:
            return 0.0, False, True, None
        tbox = res.box[target.player_id]
        tbox.targets += 1

        comp_rate = np.clip(off.comp_rate * deff.def_pass_adj ** 0.25, 0.45, 0.78)
        if side == "home":
            comp_rate += self.HOME_COMP_EDGE
        if self.rng.random() > comp_rate:
            return 0.0, False, True, None

        mean_yds = target.yards_per_catch * deff.def_pass_adj ** 0.4
        if side == "home":
            mean_yds *= self.HOME_YARDS_EDGE
        yards = float(self.rng.lognormal(np.log(max(mean_yds, 3.0)) - 0.32, 0.8))
        yards = min(yards, 99.0)
        tbox.receptions += 1
        tbox.rec_yards += yards
        box.pass_yards += yards
        return yards, False, False, target

    def _resolve_run(self, side: str, dside: str) -> tuple[float, bool, SimPlayer | None]:
        off, deff = self.profiles[side], self.profiles[dside]
        rusher = self._weighted_pick(off.players, "carry_share")
        if rusher is None:
            return 0.0, False, None
        if self.rng.random() < 0.010:  # fumble lost
            return 0.0, True, None
        mean = rusher.yards_per_carry * deff.def_rush_adj ** 0.4
        if side == "home":
            mean *= self.HOME_YARDS_EDGE
        if self.rng.random() < 0.07:   # breakaway tail
            yards = float(mean + self.rng.exponential(14.0))
        else:
            yards = float(self.rng.normal(mean, 3.2))
        yards = float(np.clip(yards, -6.0, 99.0))
        box = self.results[side].box[rusher.player_id]
        box.carries += 1
        box.rush_yards += yards
        return yards, False, rusher

    # ---------------- game loop ----------------

    def play_game(self) -> SimResult:
        rng = self.rng
        score = {"home": 0, "away": 0}
        receives_second_half = "home" if rng.random() < 0.5 else "away"
        offense = "away" if receives_second_half == "home" else "home"

        for half in (1, 2):
            if half == 2:
                offense = receives_second_half
            sec_left = 2 * QUARTER_SEC
            yardline, down, ydstogo = float(KICKOFF_YARDLINE), 1, 10.0
            drive_open = True

            while sec_left > 0:
                side, dside = offense, ("home" if offense == "away" else "away")
                res = self.results[side]
                if drive_open:
                    res.drives += 1
                    drive_open = False
                diff = score[side] - score[dside]

                if down == 4:
                    choice = self._fourth_down(yardline, ydstogo, diff, sec_left)
                    if choice == "fg":
                        kick_dist = (100 - yardline) + 17
                        make_p = float(np.clip(1.02 - (kick_dist - 18) * 0.011, 0.35, 0.99))
                        made = rng.random() < make_p
                        res.drive_outcomes["fg" if made else "downs"] += 1
                        if made:
                            score[side] += 3
                        sec_left -= 20
                        offense = dside
                        yardline = float(KICKOFF_YARDLINE if made else max(20.0, 100 - yardline))
                        down, ydstogo, drive_open = 1, 10.0, True
                        continue
                    if choice == "punt":
                        res.drive_outcomes["punt"] += 1
                        sec_left -= 25
                        offense = dside
                        yardline = float(np.clip(100 - (yardline + PUNT_NET + rng.normal(0, 6)), 10, 80))
                        down, ydstogo, drive_open = 1, 10.0, True
                        continue
                    # else: go for it — falls through to a normal snap

                # ---- a snap happens ----
                for p in self._snap_players(side):
                    res.box[p.player_id].snaps += 1
                res.plays += 1
                res.plays_by_down[down] += 1

                is_pass = rng.random() < self._pass_prob(side, down, ydstogo, diff, sec_left)
                if is_pass:
                    res.pass_plays += 1
                    res.pass_by_down[down] += 1
                    yards, turnover, incomplete, toucher = self._resolve_pass(side, dside)
                else:
                    res.run_plays += 1
                    yards, turnover, toucher = self._resolve_run(side, dside)
                    incomplete = False

                # clock: hurry-up trailing late, milk it when protecting a lead
                if incomplete:
                    runoff = self.base_runoff * 0.5
                elif sec_left < 240 and diff < 0:
                    runoff = self.base_runoff * 0.7
                elif sec_left < 300 and diff > 0:
                    runoff = self.base_runoff * 1.4
                else:
                    runoff = self.base_runoff
                sec_left -= runoff

                if turnover:
                    res.drive_outcomes["turnover"] += 1
                    offense = dside
                    yardline = float(np.clip(100 - (yardline + yards), 5, 95))
                    down, ydstogo, drive_open = 1, 10.0, True
                    continue

                yardline += yards
                if yardline >= 100:  # touchdown (+ automatic XP)
                    res.drive_outcomes["td"] += 1
                    self._credit_td(side, is_pass, toucher)
                    score[side] += 7
                    offense = dside
                    yardline, down, ydstogo, drive_open = float(KICKOFF_YARDLINE), 1, 10.0, True
                    continue
                if yardline <= 0:    # safety
                    score[dside] += 2
                    offense = dside
                    yardline, down, ydstogo, drive_open = float(KICKOFF_YARDLINE), 1, 10.0, True
                    continue

                ydstogo -= yards
                if ydstogo <= 0:
                    down, ydstogo = 1, min(10.0, 100 - yardline)
                else:
                    down += 1
                    if down > 4:
                        res.drive_outcomes["downs"] += 1
                        offense = dside
                        yardline = float(np.clip(100 - yardline, 5, 95))
                        down, ydstogo, drive_open = 1, 10.0, True

            # half ended mid-drive
            if not drive_open:
                self.results[offense].drive_outcomes["end_half"] += 1

        # single sudden-score OT possession on a tie
        if score["home"] == score["away"]:
            side = "home" if rng.random() < 0.5 else "away"
            r = rng.random()
            if r < 0.42:
                score[side] += 6
            elif r < 0.60:
                score["home" if side == "away" else "away"] += 3

        self.results["home"].points = score["home"]
        self.results["away"].points = score["away"]
        return SimResult(home=self.results["home"], away=self.results["away"])

    def _credit_td(self, side: str, was_pass: bool, toucher: SimPlayer | None) -> None:
        """Attribute the scoring play's TD to whoever touched the ball."""
        if toucher is None:
            return
        res = self.results[side]
        if was_pass:
            qb = next(p for p in self.profiles[side].players if p.position == "QB")
            res.box[qb.player_id].pass_tds += 1
            res.box[toucher.player_id].rec_tds += 1
        else:
            res.box[toucher.player_id].rush_tds += 1


def simulate_game(home: TeamProfile, away: TeamProfile, rng: np.random.Generator) -> SimResult:
    """One replicate of home vs away."""
    return _Sim(home, away, rng).play_game()

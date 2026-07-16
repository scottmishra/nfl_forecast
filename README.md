# 🏈 Gameday Edge — NFL Game-Day Forecaster

Per-player, per-position probabilistic forecasts for every game on the slate,
served through a primetime-broadcast-style dashboard. Each offensive skill
player (QB / RB / WR / TE) gets quantile forecasts (p10 floor → p90 ceiling)
for their position's stat lines, conditioned on:

- **Player form** — trailing 3- and 8-game rolling stats and usage (targets, carries, attempts)
- **Opponent quality** — rolling *defense-vs-position* fantasy points allowed, points allowed overall
- **Location** — home/away, rest days, travel distance (great-circle), altitude
- **Weather** — temperature, wind, dome/retractable-roof handling, via [Open-Meteo](https://open-meteo.com/) (free, no key)
- **Social buzz** — a pluggable sentiment/hype signal (neutral by default; Reddit provider included)

Historical data comes from [nflverse](https://github.com/nflverse/nflverse-data)
public releases (the same source `nfl_data_py` wraps).

![stack](https://img.shields.io/badge/python-3.10%2B-blue) ![gpu](https://img.shields.io/badge/GPU-8GB%20VRAM%20is%20plenty-green)

## Quickstart

```bash
pip install -e .            # core (CPU) install
gameday demo                # synthetic league, trains + forecasts offline
gameday serve               # open http://localhost:8000
```

`gameday demo` needs **no network at all** — it generates three synthetic
seasons with realistic latent structure, trains the models, and forecasts an
upcoming 16-game slate so you can explore the full product immediately.

For real data:

```bash
gameday forecast --seasons 2019,2020,2021,2022,2023,2024,2025
gameday serve
```

## Modeling

**Primary engine — LightGBM quantile ensemble.** One booster per
(position, stat, quantile) with pinball-loss validation on the held-out most
recent season. Trains the whole slate of models in ~2 minutes on CPU;
non-crossing quantiles are enforced at predict time by per-row sorting, and
all forecasts are clipped non-negative.

**GPU engine — multi-quantile MLP (`--engine neural`).** One network per
position predicts every (stat × quantile) head jointly with pinball loss, so
stats share representation (a wind signal learned on passing yards transfers
to passing TDs). ~1M parameters — it uses well under 1 GB of VRAM, so an 8 GB
card is far more than enough headroom. Install with `pip install -e '.[gpu]'`.
The train/predict interface is identical to the GBM engine; swapping in a
temporal architecture (PatchTST / TFT via `neuralforecast` or
`pytorch-forecasting`) is a drop-in replacement if you want sequence modeling
over raw game logs.

**Game simulation — discrete-event engine (`gameday/sim/`).** Beyond stat
lines, every matchup is simulated play-by-play N times (default 500): game
state is (quarter, clock, down, distance, field position, score, possession);
a situational play-calling policy (league curves + each team's calibrated
lean) picks pass/run/FG/punt/go-for-it; calibrated distributions resolve
outcomes; a pace-aware clock model advances time, with hurry-up and
clock-milking behavior. Team profiles (pace, pass rate, efficiency, defensive
adjustments, player usage shares and personnel snap weights) are estimated
from each team's trailing 8 games with shrinkage toward league means.
Aggregates per game: win probability, score distributions, expected play
calls by down, drive-outcome shares, and per-player expected snap counts /
touches with p10–p90 ranges. Simplifications by design: no penalties,
2-point tries, or onside kicks (structural outputs — play mix, snaps, win
prob — are robust to these). Tune depth with `--sims N` (0 disables).

**Leakage discipline.** Every rolling feature is shifted one game — a row
only ever sees information available before kickoff. Tests assert this
(`tests/test_features.py::test_features_no_leakage`).

## Social buzz

Free real-time Twitter/X access no longer exists, so buzz is an interface,
not a hard dependency: any callable mapping `(players, season, week) → score
in [-1, 1]` can register as a provider (`gameday/data/social.py`). Shipped
providers:

- `neutral` (default) — 0.0 for everyone; models train fine without buzz
- `reddit` — mention-volume z-score from r/fantasyfootball + r/nfl hot posts
  (`pip install -e '.[social]'` + PRAW credentials in env)

```bash
gameday forecast --buzz reddit
```

## The dashboard

- **Slate view** — every game as a card with team-color rails, venue,
  weather chips (wind/cold flagged), and the three headline projections
- **Matchup view** — team-color hero banner, then every player as a card of
  quantile strips: p10–p90 track, p25–p75 band, white median tick, hover
  tooltip with the full five-number summary
- **Simulation tab** — projected median score, win-probability bar, points
  floor/ceiling strips, pass/run mix by down, drive-outcome shares, and
  expected snap counts per player with p10–p90 whiskers
- **Player search** — jump straight to any player's matchup

No build step: it's a hand-rolled SPA (vanilla JS + CSS) served by FastAPI.
The categorical palette (position badges) is colorblind-validated against the
dark surface; identity is never encoded by color alone.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/slate` | upcoming games + venue/weather + headliners |
| `GET /api/game/{game_id}` | both teams' full player forecasts |
| `GET /api/game/{game_id}/sim` | Monte Carlo aggregates: win prob, play calls, snaps |
| `GET /api/players?q=` | player search across the slate |
| `GET /api/health` | liveness |

## Layout

```
gameday/
  config.py        seasons, positions/stats, quantiles, model params
  pipeline.py      data -> features -> train -> forecast orchestration
  cli.py           gameday demo | forecast | serve
  data/
    nflverse.py    weekly player stats + schedules (cached parquet)
    weather.py     Open-Meteo archive & forecast client, dome-aware
    teams.py       stadium geo/roof/altitude/colors + travel distance
    social.py      pluggable buzz providers (neutral, reddit)
    demo.py        synthetic league generator (offline demo/tests)
  features/build.py  leakage-safe feature matrix + per-position columns
  models/
    quantile_gbm.py  LightGBM quantile ensemble (primary)
    neural.py        torch multi-quantile MLP (optional GPU path)
  sim/
    calibrate.py     team profiles: pace, tendencies, efficiency, personnel
    engine.py        discrete-event play-by-play simulator
    run.py           Monte Carlo runner + slate aggregation
  api/server.py    FastAPI backend + static dashboard
web/               the dashboard (index.html / styles.css / app.js)
tests/             feature-leakage, quantile-sanity, end-to-end pipeline
```

## Tests

```bash
python -m pytest
```

Covers: demo-league shape, rolling-feature leakage, upcoming-slate scaffolding,
dome weather neutralization, quantile ordering/non-negativity, and an
end-to-end pipeline run.

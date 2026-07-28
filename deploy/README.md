# Model deployment: train big, deploy small

Training runs on a workstation (GPU/fast CPU); the Raspberry Pi only ever
downloads finished model bundles. `models.json` in this directory is the
**deploy pointer** — the single source of truth for which bundle a Pi runs.

## Pointer format (`models.json`)

```json
{
  "schema": 1,
  "version": "models-20260727-193000",
  "url": "https://github.com/<owner>/<repo>/releases/download/<version>/model_bundle-<version>.tar.gz",
  "sha256": "<sha256 of the tar.gz>",
  "published_at": "2026-07-27T19:31:02+00:00"
}
```

- `url` may also be a filesystem path (absolute, or relative to this
  directory) — used for offline tests and LAN installs.
- All fields `null` means nothing is published yet; `gameday models sync`
  refuses to run and `gameday refresh` continues with whatever bundle is
  already installed.

## Flow

1. **Train + gate + pack** (workstation):
   `gameday train --seasons 2022,2023,2024 --engine gbm`
   Trains into `artifacts/models/current`, backtests the latest completed
   season, and — if the gate passes — packs
   `artifacts/bundles/model_bundle-<version>.tar.gz`.
2. **Publish** (workstation, manual for now):
   `gameday models publish` prints the exact `gh release create` command and
   the pointer JSON to commit. It never executes them — run the printed
   commands yourself after review.
3. **Sync** (Pi): `gameday models sync` reads this pointer, downloads the
   bundle, verifies its sha256 and per-file manifest hashes, and installs it
   as `artifacts/models/current` (the prior set becomes `previous/`).
   The nightly `gameday refresh` does this automatically (best-effort: a
   failed sync never kills a refresh that already has installed models).
4. **Rollback** (Pi): `gameday models rollback` swaps `previous/` back in.

Bundle integrity: `manifest.json` inside the tar lists every file with its
sha256; installs verify each file and refuse path traversal, mismatched
hashes, wrong `feature_schema_version`, or a missing engine dependency.

## Pi install: nightly refresh timer

The units in `systemd/` are **user** units (same pattern as the fleet's
`claude-rc.service`) and assume the repo lives at `~/nfl_forecast` with its
venv at `~/nfl_forecast/.venv`.

```bash
mkdir -p ~/.config/systemd/user
cp ~/nfl_forecast/deploy/systemd/gameday-refresh.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gameday-refresh.timer

# verify
systemctl --user list-timers gameday-refresh.timer   # next fire time
systemctl --user start gameday-refresh.service       # run one refresh now
journalctl --user -u gameday-refresh.service -n 50   # inspect the last run
cat ~/nfl_forecast/artifacts/forecasts/refresh_status.json
```

Notes:

- Over SSH the user bus may not be up; prefix commands with
  `XDG_RUNTIME_DIR=/run/user/1000` (adjust the uid to `id -u`) if
  `systemctl --user` complains about the bus.
- User timers only run while the user has a session unless lingering is on:
  `loginctl show-user $USER | grep Linger` — if `Linger=no`, enable it with
  `sudo loginctl enable-linger $USER` so the timer fires after reboots
  without a login.
- The timer fires at 10:30 UTC (see the comment in the unit: after
  nflverse's overnight-ET publishing) with up to 10 minutes of jitter;
  `Persistent=true` catches up missed runs after downtime.

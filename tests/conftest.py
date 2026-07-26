"""Isolate test outputs from the repo's shipped artifacts.

The pipeline/backtest tests write models, forecasts, and sim artifacts. This
redirects gameday's data/artifacts roots to a temp directory BEFORE any
gameday module is imported (config reads these env vars at import time), so
running the suite never dirties artifacts/forecasts/ committed in git.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="gameday-tests-")
os.environ["GAMEDAY_DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["GAMEDAY_ARTIFACTS_DIR"] = os.path.join(_tmp, "artifacts")

"""Social-media "buzz" signal: pluggable sentiment providers.

Free real-time Twitter/X access no longer exists, so the buzz feature is an
interface: register any provider that maps (player, week) -> a score in
[-1, 1]. Ships with a neutral provider (default) and an optional Reddit
provider (pip install gameday[social] + PRAW credentials). The feature
pipeline treats missing buzz as 0.0, so models train fine without it.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import pandas as pd

log = logging.getLogger(__name__)


class BuzzProvider(Protocol):
    def __call__(self, players: pd.DataFrame, season: int, week: int) -> pd.Series:
        """Return a buzz score in [-1, 1] indexed like `players` (player_id rows)."""
        ...


_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_provider(name: str = "neutral") -> Callable:
    return _REGISTRY[name]


@register("neutral")
def neutral_buzz(players: pd.DataFrame, season: int, week: int) -> pd.Series:
    return pd.Series(0.0, index=players.index)


@register("reddit")
def reddit_buzz(players: pd.DataFrame, season: int, week: int) -> pd.Series:
    """Mention-volume z-score from r/fantasyfootball + r/nfl hot posts.

    Requires PRAW credentials in env (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET /
    REDDIT_USER_AGENT). Score = tanh of the player's share of name-mentions,
    which behaves as hype/volume rather than polarity — useful for spotting
    breakout chatter and injury noise alike.
    """
    try:
        import praw  # type: ignore
    except ImportError:
        log.warning("praw not installed; falling back to neutral buzz")
        return neutral_buzz(players, season, week)

    import os

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "gameday-forecaster"),
    )
    text = " ".join(
        f"{s.title} {getattr(s, 'selftext', '')}"
        for sub in ("fantasyfootball", "nfl")
        for s in reddit.subreddit(sub).hot(limit=150)
    ).lower()

    counts = players["player_display_name"].str.lower().map(text.count).astype(float)
    if counts.sum() == 0:
        return neutral_buzz(players, season, week)
    z = (counts - counts.mean()) / (counts.std() or 1.0)
    import numpy as np

    return pd.Series(np.tanh(z / 2.0), index=players.index)

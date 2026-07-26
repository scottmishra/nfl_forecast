"""Game-site weather from Open-Meteo (free, no API key).

Historical games use the archive API; upcoming games use the forecast API.
Dome games are short-circuited to neutral indoor conditions.
"""

from __future__ import annotations

import datetime as dt
import logging

import httpx

from gameday.data.teams import TEAMS

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

INDOOR = {"temp_c": 21.0, "wind_kph": 0.0, "precip_mm": 0.0, "is_dome": 1}


def game_weather(home_team: str, kickoff: dt.datetime) -> dict:
    """Hourly conditions at the stadium nearest the kickoff hour.

    Returns temp_c / wind_kph / precip_mm / is_dome; falls back to mild
    outdoor defaults when the API is unreachable so the pipeline never blocks.
    """
    venue = TEAMS[home_team]
    if venue["roof"] == "dome":
        return dict(INDOOR)

    is_past = kickoff.date() < dt.date.today()
    url = ARCHIVE_URL if is_past else FORECAST_URL
    params = {
        "latitude": venue["lat"],
        "longitude": venue["lon"],
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "start_date": kickoff.date().isoformat(),
        "end_date": kickoff.date().isoformat(),
        "timezone": "auto",
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            hourly = resp.json()["hourly"]
        idx = min(kickoff.hour, len(hourly["temperature_2m"]) - 1)
        return {
            "temp_c": float(hourly["temperature_2m"][idx]),
            "wind_kph": float(hourly["wind_speed_10m"][idx]),
            "precip_mm": float(hourly["precipitation"][idx]),
            "is_dome": 0,
        }
    except Exception as exc:  # network-optional: never let weather block a forecast
        log.warning("weather lookup failed for %s (%s); using neutral outdoor defaults", home_team, exc)
        return {"temp_c": 15.0, "wind_kph": 8.0, "precip_mm": 0.0, "is_dome": 0}

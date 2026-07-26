"""Static NFL team/stadium reference: geography, venue type, and brand colors.

Latitude/longitude power weather lookups and travel-distance features;
dome/retractable flags gate whether weather affects the forecast at all.
"""

from __future__ import annotations

import math

# abbr: (team name, stadium, city, lat, lon, roof, surface, altitude_m, primary, secondary)
# roof: "outdoor" | "dome" | "retractable"
TEAMS: dict[str, dict] = {
    "ARI": dict(name="Arizona Cardinals", stadium="State Farm Stadium", city="Glendale, AZ", lat=33.5276, lon=-112.2626, roof="retractable", surface="grass", altitude=331, primary="#97233F", secondary="#FFB612"),
    "ATL": dict(name="Atlanta Falcons", stadium="Mercedes-Benz Stadium", city="Atlanta, GA", lat=33.7554, lon=-84.4008, roof="retractable", surface="turf", altitude=306, primary="#A71930", secondary="#000000"),
    "BAL": dict(name="Baltimore Ravens", stadium="M&T Bank Stadium", city="Baltimore, MD", lat=39.2780, lon=-76.6227, roof="outdoor", surface="grass", altitude=6, primary="#241773", secondary="#9E7C0C"),
    "BUF": dict(name="Buffalo Bills", stadium="Highmark Stadium", city="Orchard Park, NY", lat=42.7738, lon=-78.7870, roof="outdoor", surface="turf", altitude=180, primary="#00338D", secondary="#C60C30"),
    "CAR": dict(name="Carolina Panthers", stadium="Bank of America Stadium", city="Charlotte, NC", lat=35.2258, lon=-80.8528, roof="outdoor", surface="turf", altitude=229, primary="#0085CA", secondary="#101820"),
    "CHI": dict(name="Chicago Bears", stadium="Soldier Field", city="Chicago, IL", lat=41.8623, lon=-87.6167, roof="outdoor", surface="grass", altitude=181, primary="#0B162A", secondary="#C83803"),
    "CIN": dict(name="Cincinnati Bengals", stadium="Paycor Stadium", city="Cincinnati, OH", lat=39.0954, lon=-84.5160, roof="outdoor", surface="turf", altitude=147, primary="#FB4F14", secondary="#000000"),
    "CLE": dict(name="Cleveland Browns", stadium="Huntington Bank Field", city="Cleveland, OH", lat=41.5061, lon=-81.6995, roof="outdoor", surface="grass", altitude=177, primary="#311D00", secondary="#FF3C00"),
    "DAL": dict(name="Dallas Cowboys", stadium="AT&T Stadium", city="Arlington, TX", lat=32.7473, lon=-97.0945, roof="retractable", surface="turf", altitude=168, primary="#003594", secondary="#869397"),
    "DEN": dict(name="Denver Broncos", stadium="Empower Field at Mile High", city="Denver, CO", lat=39.7439, lon=-105.0201, roof="outdoor", surface="grass", altitude=1609, primary="#FB4F14", secondary="#002244"),
    "DET": dict(name="Detroit Lions", stadium="Ford Field", city="Detroit, MI", lat=42.3400, lon=-83.0456, roof="dome", surface="turf", altitude=183, primary="#0076B6", secondary="#B0B7BC"),
    "GB":  dict(name="Green Bay Packers", stadium="Lambeau Field", city="Green Bay, WI", lat=44.5013, lon=-88.0622, roof="outdoor", surface="grass", altitude=195, primary="#203731", secondary="#FFB612"),
    "HOU": dict(name="Houston Texans", stadium="NRG Stadium", city="Houston, TX", lat=29.6847, lon=-95.4107, roof="retractable", surface="turf", altitude=15, primary="#03202F", secondary="#A71930"),
    "IND": dict(name="Indianapolis Colts", stadium="Lucas Oil Stadium", city="Indianapolis, IN", lat=39.7601, lon=-86.1639, roof="retractable", surface="turf", altitude=218, primary="#002C5F", secondary="#A2AAAD"),
    "JAX": dict(name="Jacksonville Jaguars", stadium="EverBank Stadium", city="Jacksonville, FL", lat=30.3240, lon=-81.6373, roof="outdoor", surface="grass", altitude=5, primary="#006778", secondary="#D7A22A"),
    "KC":  dict(name="Kansas City Chiefs", stadium="GEHA Field at Arrowhead", city="Kansas City, MO", lat=39.0489, lon=-94.4839, roof="outdoor", surface="grass", altitude=266, primary="#E31837", secondary="#FFB81C"),
    "LAC": dict(name="Los Angeles Chargers", stadium="SoFi Stadium", city="Inglewood, CA", lat=33.9535, lon=-118.3392, roof="dome", surface="turf", altitude=30, primary="#0080C6", secondary="#FFC20E"),
    "LAR": dict(name="Los Angeles Rams", stadium="SoFi Stadium", city="Inglewood, CA", lat=33.9535, lon=-118.3392, roof="dome", surface="turf", altitude=30, primary="#003594", secondary="#FFA300"),
    "LV":  dict(name="Las Vegas Raiders", stadium="Allegiant Stadium", city="Las Vegas, NV", lat=36.0909, lon=-115.1833, roof="dome", surface="grass", altitude=610, primary="#000000", secondary="#A5ACAF"),
    "MIA": dict(name="Miami Dolphins", stadium="Hard Rock Stadium", city="Miami Gardens, FL", lat=25.9580, lon=-80.2389, roof="outdoor", surface="grass", altitude=3, primary="#008E97", secondary="#FC4C02"),
    "MIN": dict(name="Minnesota Vikings", stadium="U.S. Bank Stadium", city="Minneapolis, MN", lat=44.9735, lon=-93.2575, roof="dome", surface="turf", altitude=253, primary="#4F2683", secondary="#FFC62F"),
    "NE":  dict(name="New England Patriots", stadium="Gillette Stadium", city="Foxborough, MA", lat=42.0909, lon=-71.2643, roof="outdoor", surface="turf", altitude=89, primary="#002244", secondary="#C60C30"),
    "NO":  dict(name="New Orleans Saints", stadium="Caesars Superdome", city="New Orleans, LA", lat=29.9511, lon=-90.0812, roof="dome", surface="turf", altitude=1, primary="#D3BC8D", secondary="#101820"),
    "NYG": dict(name="New York Giants", stadium="MetLife Stadium", city="East Rutherford, NJ", lat=40.8135, lon=-74.0745, roof="outdoor", surface="turf", altitude=2, primary="#0B2265", secondary="#A71930"),
    "NYJ": dict(name="New York Jets", stadium="MetLife Stadium", city="East Rutherford, NJ", lat=40.8135, lon=-74.0745, roof="outdoor", surface="turf", altitude=2, primary="#125740", secondary="#FFFFFF"),
    "PHI": dict(name="Philadelphia Eagles", stadium="Lincoln Financial Field", city="Philadelphia, PA", lat=39.9008, lon=-75.1675, roof="outdoor", surface="grass", altitude=12, primary="#004C54", secondary="#A5ACAF"),
    "PIT": dict(name="Pittsburgh Steelers", stadium="Acrisure Stadium", city="Pittsburgh, PA", lat=40.4468, lon=-80.0158, roof="outdoor", surface="grass", altitude=224, primary="#FFB612", secondary="#101820"),
    "SEA": dict(name="Seattle Seahawks", stadium="Lumen Field", city="Seattle, WA", lat=47.5952, lon=-122.3316, roof="outdoor", surface="turf", altitude=5, primary="#002244", secondary="#69BE28"),
    "SF":  dict(name="San Francisco 49ers", stadium="Levi's Stadium", city="Santa Clara, CA", lat=37.4033, lon=-121.9694, roof="outdoor", surface="grass", altitude=8, primary="#AA0000", secondary="#B3995D"),
    "TB":  dict(name="Tampa Bay Buccaneers", stadium="Raymond James Stadium", city="Tampa, FL", lat=27.9759, lon=-82.5033, roof="outdoor", surface="grass", altitude=8, primary="#D50A0A", secondary="#34302B"),
    "TEN": dict(name="Tennessee Titans", stadium="Nissan Stadium", city="Nashville, TN", lat=36.1665, lon=-86.7713, roof="outdoor", surface="grass", altitude=128, primary="#0C2340", secondary="#4B92DB"),
    "WAS": dict(name="Washington Commanders", stadium="Northwest Stadium", city="Landover, MD", lat=38.9077, lon=-76.8645, roof="outdoor", surface="grass", altitude=61, primary="#5A1414", secondary="#FFB612"),
}


def travel_km(home: str, away: str) -> float:
    """Great-circle distance the away team travels, in km."""
    a, b = TEAMS[home], TEAMS[away]
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))

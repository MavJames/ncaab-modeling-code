from __future__ import annotations

import time
from typing import Iterable, Optional

import pandas as pd
import requests

CONF_MAP = {
    "Atlantic Coast Conference": "acc",
    "America East Conference": "america-east",
    "American Conference": "american",
    "Atlantic 10 Conference": "atlantic-10",
    "Atlantic Sun Conference": "atlantic-sun",
    "Big 12 Conference": "big-12",
    "Big East Conference": "big-east",
    "Big Sky Conference": "big-sky",
    "Big South Conference": "big-south",
    "Big Ten Conference": "big-ten",
    "Big West Conference": "big-west",
    "Coastal Athletic Association": "coastal",
    "Conference USA": "cusa",
    "Horizon League": "horizon",
    "Ivy League": "ivy",
    "Metro Atlantic Athletic Conference": "maac",
    "Mid-American Conference": "mac",
    "Mid-Eastern Athletic Conference": "meac",
    "Missouri Valley Conference": "mvc",
    "Mountain West Conference": "mwc",
    "NEC": "nec",
    "Ohio Valley Conference": "ovc",
    "Patriot League": "patriot",
    "Southeastern Conference": "sec",
    "Southern Conference": "southern",
    "Southland Conference": "southland",
    "Summit League": "summit",
    "Sun Belt Conference": "sun-belt",
    "Southwest Athletic Conference": "swac",
    "Western Athletic Conference": "wac",
    "West Coast Conference": "wcc",
}

BASE = "https://www.sports-reference.com/cbb/conferences/{}/men/schools.html"


def fetch_conference_teams(
    full_name: str,
    slug: str,
    *,
    to_year: int = 2026,
    sleep_seconds: float = 2.0,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch a single conference's teams for a given year."""
    url = BASE.format(slug)
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

    sess = session or requests.Session()
    response = sess.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    df = pd.read_html(response.text, attrs={"id": "schools"})[0]
    df["To"] = pd.to_numeric(df["To"], errors="coerce")
    df = df[df["To"] == to_year]

    df = df.rename(columns={"School": "team_name"})
    df["conference"] = full_name
    df["conference_slug"] = slug

    time.sleep(sleep_seconds)
    return df[["team_name", "conference", "conference_slug"]]


def build_conference_assignments(
    *,
    to_year: int = 2026,
    sleep_seconds: float = 2.0,
    conference_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build a dataframe of teams with conference assignments."""
    rows: list[pd.DataFrame] = []
    conf_map = conference_map or CONF_MAP

    for full_name, slug in conf_map.items():
        rows.append(
            fetch_conference_teams(
                full_name,
                slug,
                to_year=to_year,
                sleep_seconds=sleep_seconds,
            )
        )

    return pd.concat(rows, ignore_index=True)


def save_conference_assignments(
    output_path: str,
    *,
    to_year: int = 2026,
    sleep_seconds: float = 2.0,
    conference_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Create and save the conference assignments CSV."""
    df = build_conference_assignments(
        to_year=to_year,
        sleep_seconds=sleep_seconds,
        conference_map=conference_map,
    )
    df.to_csv(output_path, index=False)
    return df

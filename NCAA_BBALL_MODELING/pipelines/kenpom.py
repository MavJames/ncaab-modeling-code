from __future__ import annotations

import os
from typing import Optional

import pandas as pd

try:
    from kenpompy.utils import login
    import kenpompy.FanMatch as kp
except ImportError:  # pragma: no cover - optional dependency
    login = None
    kp = None

try:
    from NCAA_BBALL_MODELING.gamelog_scraping import get_schools
except ImportError:
    from gamelog_scraping import get_schools


def login_kenpom(username: Optional[str] = None, password: Optional[str] = None):
    """Login to KenPom. Username/password can come from env vars."""
    if login is None:
        raise ImportError("kenpompy is not installed")

    user = username or os.getenv("KENPOM_USER")
    pw = password or os.getenv("KENPOM_PASS")
    if not user or not pw:
        raise ValueError("KenPom credentials required (args or env vars)")

    return login(user, pw)


def fetch_fanmatch(
    match_date: str,
    *,
    browser=None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch KenPom FanMatch data for a date (YYYY-MM-DD)."""
    if kp is None:
        raise ImportError("kenpompy is not installed")

    if browser is None:
        browser = login_kenpom(username=username, password=password)

    fanmatch = kp.FanMatch(browser, match_date)
    return fanmatch.fm_df


def build_team_ids(year: int = 2026) -> pd.DataFrame:
    """Build team_id lookup from Sports Reference schools list."""
    schools = get_schools(year)
    teams_df = (
        pd.DataFrame(
            [{"school_slug": slug, "school_name": name} for slug, name in schools.items()]
        )
        .sort_values("school_name")
        .reset_index(drop=True)
    )

    teams_df["team_id"] = teams_df.index + 1
    return teams_df[["team_id", "school_name", "school_slug"]]


def save_team_ids(output_path: str, year: int = 2026) -> pd.DataFrame:
    """Save team_ids CSV to disk."""
    df = build_team_ids(year)
    df.to_csv(output_path, index=False)
    return df

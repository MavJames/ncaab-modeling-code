from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from pathlib import Path
from dotenv import load_dotenv


load_dotenv(Path("/Users/mavinjames/Projects/Basketball_Modeling/NCAA_BBALL_MODELING/.env"))


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

try:
    from NCAA_BBALL_MODELING.utils import _resolve_base_dir
except ImportError:
    from utils import _resolve_base_dir


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


def _map_team_name(name: str, team_name_map: Optional[dict[str, str]] = None) -> str:
    if pd.isna(name):
        return ""
    value = str(name).strip()
    if team_name_map:
        return team_name_map.get(value, value)
    return value


def enrich_fanmatch_predictions(fanmatch_df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed score/spread columns used in KenPom notebook work."""
    df = fanmatch_df.copy()
    if "PredictedScore" in df.columns:
        split_scores = df["PredictedScore"].astype(str).str.split("-", n=1, expand=True)
        df["Winner_Score"] = pd.to_numeric(split_scores[0], errors="coerce")
        df["Loser_Score"] = pd.to_numeric(split_scores[1], errors="coerce")
        df["KenPom_spread"] = df["Winner_Score"] - df["Loser_Score"]

    # Fallback for cases where score parsing fails or PredictedScore is absent.
    if "KenPom_spread" not in df.columns and "PredictedMOV" in df.columns:
        df["KenPom_spread"] = pd.to_numeric(df["PredictedMOV"], errors="coerce")
    return df


def _build_kenpom_merge_keys(
    fanmatch_df: pd.DataFrame,
    match_date: str,
    team_name_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    enriched = enrich_fanmatch_predictions(fanmatch_df)
    if "KenPom_spread" not in enriched.columns:
        raise KeyError(
            "KenPom_spread missing from FanMatch data. Expected PredictedScore or PredictedMOV."
        )

    winner = enriched["PredictedWinner"].map(
        lambda x: _map_team_name(x, team_name_map=team_name_map)
    )
    loser = enriched["PredictedLoser"].map(
        lambda x: _map_team_name(x, team_name_map=team_name_map)
    )

    merged = pd.DataFrame(
        {
            "date": pd.to_datetime(match_date),
            "team_a": winner.where(winner <= loser, loser),
            "team_b": loser.where(winner <= loser, winner),
            "kenpom_favorite": winner,
            "KenPom_spread": enriched["KenPom_spread"],
        }
    )
    return merged.drop_duplicates(["date", "team_a", "team_b"])


def load_name_map(map_path: str | Path) -> dict[str, str]:
    """Load a Team->school_name map from CSV."""
    map_df = pd.read_csv(map_path)
    if {"Team", "school_name"}.issubset(map_df.columns):
        left_col, right_col = "Team", "school_name"
    elif {"kenpom_name", "sportsref_name"}.issubset(map_df.columns):
        left_col, right_col = "kenpom_name", "sportsref_name"
    else:
        raise ValueError(
            "Name map CSV must contain Team/school_name or kenpom_name/sportsref_name columns."
        )

    clean = map_df[[left_col, right_col]].dropna()
    clean[left_col] = clean[left_col].astype(str).str.strip()
    clean[right_col] = clean[right_col].astype(str).str.strip()
    clean = clean.drop_duplicates(subset=[left_col], keep="last")
    return dict(zip(clean[left_col], clean[right_col]))


def update_kenpom_history(
    match_date: str,
    *,
    history_path: Optional[str | Path] = None,
    name_map_path: Optional[str | Path] = None,
    browser=None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Path:
    """Fetch KenPom for a date and upsert into persistent history CSV."""
    base_dir = _resolve_base_dir()
    if history_path is None:
        history_path = base_dir / "data" / "kenpom_spreads_history.csv"
    history_path = Path(history_path)

    team_name_map = None
    if name_map_path:
        team_name_map = load_name_map(name_map_path)

    fanmatch_df = fetch_fanmatch(
        match_date,
        browser=browser,
        username=username,
        password=password,
    )
    daily = _build_kenpom_merge_keys(
        fanmatch_df,
        match_date=match_date,
        team_name_map=team_name_map,
    )

    if history_path.exists():
        history = pd.read_csv(history_path)
        if "date" in history.columns:
            history["date"] = pd.to_datetime(history["date"], errors="coerce")
    else:
        history = pd.DataFrame(columns=daily.columns)

    combined = pd.concat([history, daily], ignore_index=True)
    combined = combined.drop_duplicates(["date", "team_a", "team_b"], keep="last")
    combined = combined.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)
    return history_path


def merge_kenpom_history_into_predictions(
    predictions_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge KenPom history into predictions with school->opp fallback by date."""
    pred = predictions_df.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    # Avoid pandas suffix collisions when rerunning daily merges.
    pred = pred.drop(
        columns=["KenPom_spread", "kenpom_spread_for_school", "kenpom_favorite"],
        errors="ignore",
    )

    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    if "KenPom_spread" not in history.columns and "kenpom_spread" in history.columns:
        history["KenPom_spread"] = history["kenpom_spread"]
    keep_cols = ["date", "kenpom_favorite", "KenPom_spread"]
    history = history[keep_cols].drop_duplicates(["date", "kenpom_favorite"], keep="last")

    school_match = pred.merge(
        history,
        left_on=["date", "school_name"],
        right_on=["date", "kenpom_favorite"],
        how="left",
        suffixes=("", "_school"),
    )
    opp_match = pred.merge(
        history,
        left_on=["date", "opp_name_abbr"],
        right_on=["date", "kenpom_favorite"],
        how="left",
        suffixes=("", "_opp"),
    )

    merged = school_match.copy()
    merged["kenpom_favorite"] = merged["kenpom_favorite"].where(
        merged["kenpom_favorite"].notna(),
        opp_match["kenpom_favorite"],
    )
    merged["KenPom_spread"] = merged["KenPom_spread"].where(
        merged["KenPom_spread"].notna(),
        opp_match["KenPom_spread"],
    )
    merged["kenpom_spread_for_school"] = merged["KenPom_spread"].where(
        school_match["kenpom_favorite"].notna(),
        -opp_match["KenPom_spread"],
    )
    return merged


def merge_predictions_with_kenpom_history(
    *,
    predictions_path: Optional[str | Path] = None,
    history_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Merge persistent KenPom history into predictions.csv."""
    base_dir = _resolve_base_dir()
    if predictions_path is None:
        predictions_path = base_dir / "data" / "predictions.csv"
    if history_path is None:
        history_path = base_dir / "data" / "kenpom_spreads_history.csv"
    if output_path is None:
        output_path = predictions_path

    pred = pd.read_csv(predictions_path)
    history = pd.read_csv(history_path)
    merged = merge_kenpom_history_into_predictions(pred, history)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def run_kenpom_history_update(
    match_date: str,
    *,
    predictions_path: Optional[str | Path] = None,
    history_path: Optional[str | Path] = None,
    name_map_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    browser=None,
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Path:
    """Update KenPom history for one date, then merge it into predictions.csv."""
    history_path = update_kenpom_history(
        match_date=match_date,
        history_path=history_path,
        name_map_path=name_map_path,
        browser=browser,
        username=username,
        password=password,
    )
    return merge_predictions_with_kenpom_history(
        predictions_path=predictions_path,
        history_path=history_path,
        output_path=output_path,
    )


def merge_kenpom_into_predictions(
    predictions_df: pd.DataFrame,
    fanmatch_df: pd.DataFrame,
    match_date: str,
    team_name_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Join KenPom spread into predictions by date + unordered team pair."""
    pred = predictions_df.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["school_key"] = pred["school_name"].map(
        lambda x: _map_team_name(x, team_name_map=team_name_map)
    )
    pred["opp_key"] = pred["opp_name_abbr"].map(
        lambda x: _map_team_name(x, team_name_map=team_name_map)
    )
    pred["team_a"] = pred["school_key"].where(
        pred["school_key"] <= pred["opp_key"], pred["opp_key"]
    )
    pred["team_b"] = pred["opp_key"].where(
        pred["school_key"] <= pred["opp_key"], pred["school_key"]
    )

    kp_keys = _build_kenpom_merge_keys(
        fanmatch_df,
        match_date=match_date,
        team_name_map=team_name_map,
    )
    merged = pred.merge(
        kp_keys,
        on=["date", "team_a", "team_b"],
        how="left",
    )
    merged["kenpom_spread_for_school"] = merged["KenPom_spread"].where(
        merged["school_key"] == merged["kenpom_favorite"],
        -merged["KenPom_spread"],
    )
    return merged.drop(
        columns=["school_key", "opp_key", "team_a", "team_b"],
        errors="ignore",
    )


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

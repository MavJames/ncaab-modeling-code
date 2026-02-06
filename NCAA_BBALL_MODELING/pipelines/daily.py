from __future__ import annotations

from pathlib import Path
from typing import Optional

from .engineering import run_engineering


def run_all(
    *,
    target_date: Optional[str] = None,
    season: int = 2026,
    run_update: bool = True,
    run_features: bool = True,
    base_dir: Optional[str | Path] = None,
    max_teams: Optional[int] = None,
) -> Optional[Path]:
    """Run engineering steps together. Returns features path if created."""
    return run_engineering(
        target_date=target_date,
        season=season,
        run_update=run_update,
        run_features=run_features,
        base_dir=base_dir,
        max_teams=max_teams,
    )

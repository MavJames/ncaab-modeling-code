from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from NCAA_BBALL_MODELING import utils
except ImportError:
    import utils


def update_gamelogs(
    target_date: str,
    season: int = 2026,
    input_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    max_teams: Optional[int] = None,
) -> None:
    """Update gamelogs for a specific date (YYYY-MM-DD)."""
    utils.update_gamelogs_by_date(
        target_date=target_date,
        season=season,
        input_path=input_path,
        output_path=output_path,
        max_teams=max_teams,
    )


def create_features(only_season: Optional[int] = None, base_dir: Optional[str | Path] = None) -> Path:
    """Create feature CSVs. Returns the output path."""
    utils.create_features(only_season=only_season, base_dir=base_dir)

    base_dir_resolved = Path(base_dir) if base_dir is not None else utils._resolve_base_dir()
    data_dir = base_dir_resolved / "data"
    if only_season == 2026:
        return data_dir / "2026" / "features_2026_new.csv"
    return data_dir / "merged_dataset.csv"


def run_engineering(
    *,
    target_date: Optional[str] = None,
    season: int = 2026,
    run_update: bool = True,
    run_features: bool = True,
    base_dir: Optional[str | Path] = None,
    max_teams: Optional[int] = None,
) -> Optional[Path]:
    """Run engineering steps separately or together."""
    if run_update:
        if not target_date:
            raise ValueError("target_date is required when run_update=True")
        update_gamelogs(
            target_date=target_date,
            season=season,
            max_teams=max_teams,
        )

    if run_features:
        return create_features(only_season=season, base_dir=base_dir)

    return None

from __future__ import annotations

import argparse
from pathlib import Path

from NCAA_BBALL_MODELING.pipelines.engineering import run_engineering
from NCAA_BBALL_MODELING.pipelines.modeling import run_modeling_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily NCAAB pipeline steps.")

    parser.add_argument("--season", type=int, default=2026, help="Season year.")
    parser.add_argument(
        "--date",
        dest="target_date",
        help="Target date for update step (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Skip the gamelog update step.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip the feature creation step.",
    )
    parser.add_argument(
        "--run-modeling",
        action="store_true",
        help="Run modeling and write predictions.csv.",
    )
    parser.add_argument(
        "--features-path",
        help="Path to features CSV (defaults to data/<season>/features_<season>.csv).",
    )
    parser.add_argument(
        "--training-path",
        help="Path to merged training CSV (defaults to data/merged_dataset.csv).",
    )
    parser.add_argument(
        "--predictions-path",
        help="Output path for predictions CSV (defaults to data/predictions.csv).",
    )
    parser.add_argument(
        "--max-teams",
        type=int,
        help="Only update first N teams found on --date (checker mode).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_update = not args.skip_update
    run_features = not args.skip_features

    if run_update and not args.target_date:
        raise SystemExit("--date is required unless --skip-update is set")

    features_path = run_engineering(
        target_date=args.target_date,
        season=args.season,
        run_update=run_update,
        run_features=run_features,
        max_teams=args.max_teams,
    )

    if args.run_modeling:
        run_modeling_pipeline(
            training_path=args.training_path,
            features_path=args.features_path or features_path,
            predictions_path=args.predictions_path,
            season_test=args.season,
        )


if __name__ == "__main__":
    main()

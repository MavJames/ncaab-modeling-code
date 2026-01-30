import argparse
from pathlib import Path

import pandas as pd

from update_gamelogs_by_date import update_gamelogs_for_date
from data.feature_engineering_2 import main as build_features


def main():
    parser = argparse.ArgumentParser(
        description="Update gamelogs for a specific date and rebuild features."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season year (default: 2026).",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=3,
        help="Seconds to sleep between team scrapes.",
    )
    args = parser.parse_args()

    target_date = (
        pd.Timestamp("today").normalize()
        if args.date is None
        else pd.to_datetime(args.date).normalize()
    )

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / str(args.season)
    input_path = data_dir / f"NCAAB_{args.season}_Team_Gamelogs.xlsx"

    # Overwrite the input file with updated rows for the target date.
    update_gamelogs_for_date(
        input_path=input_path,
        output_path=input_path,
        target_date=target_date,
        season=args.season,
        sleep_seconds=args.sleep,
    )

    # Rebuild features for the season after updating gamelogs.
    build_features(only_season=args.season)


if __name__ == "__main__":
    main()

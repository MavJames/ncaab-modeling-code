import argparse
from pathlib import Path
from time import sleep

import pandas as pd

from gamelog_scraping import scrape_team_gamelog

RENAME_MAP = {
    "Texas A&M–Commerce": "East Texas A&M",
    "Texas–Rio Grande Valley": "Texas-Rio Grande Valley",
    "Sam Houston State": "Sam Houston",
    "USC Upstate": "South Carolina Upstate",
    "Arkansas–Pine Bluff": "Arkansas-Pine Bluff",
    "UNLV": "Nevada-Las Vegas",
    "Prairie View A&M": "Prairie View",
    "Grambling State": "Grambling",
    "LIU": "Long Island University",
    "Loyola Chicago": "Loyola (IL)",
    "UMBC": "Maryland-Baltimore County",
    "UMass Lowell": "Massachusetts-Lowell",
    "Ole Miss": "Mississippi",
    "Texas A&M–Corpus Christi": "Texas A&M-Corpus Christi",
    "Louisiana–Monroe": "Louisiana-Monroe",
    "UT Martin": "Tennessee-Martin",
    "Illinois–Chicago": "Illinois-Chicago",
    "St. Mary's (CA)": "Saint Mary's (CA)",
    "Fairleigh Dickinson": "FDU",
    "Maryland Eastern Shore": "Maryland-Eastern Shore",
    "IUPUI": "IU Indy",
    "SMU": "Southern Methodist",
    "VCU": "Virginia Commonwealth",
}


def clean_gamelogs(df, rename_map=RENAME_MAP):
    df = df.copy()
    df["school_name"] = df["school_name"].str.replace(r"NCAA$", "", regex=True)
    df["date"] = pd.to_datetime(df["date"])
    df["opp_name_abbr"] = df["opp_name_abbr"].replace(rename_map)

    valid_schools = set(df["school_name"].unique())
    df = df[df["opp_name_abbr"].isin(valid_schools)]

    # Drop canceled games with null results except the final "next game" row per team/season
    df = df.sort_values(["season", "school_name", "date"])
    g = df.groupby(["season", "school_name"], sort=False)
    group_size = g.size().rename("group_size")
    df = df.join(group_size, on=["season", "school_name"])
    df["row_in_group"] = g.cumcount()
    is_last = df["row_in_group"] == (df["group_size"] - 1)
    df = df[~(df["team_game_result"].isna() & ~is_last)]
    df = df.drop(columns=["group_size", "row_in_group"])
    return df


def update_gamelogs_for_date(
    input_path, output_path, target_date, season, sleep_seconds=6
):
    df = pd.read_excel(input_path, index_col=False)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    target_date = pd.to_datetime(target_date).normalize()
    teams_today = df.loc[df["date"] == target_date, ["school_slug", "school_name"]]
    teams_today = teams_today.dropna().drop_duplicates()

    if teams_today.empty:
        print(f"No teams found for {target_date.date()} in {input_path}")
        return

    updated_rows = []
    for i, row in teams_today.reset_index(drop=True).iterrows():
        slug = row["school_slug"]
        name = row["school_name"]
        print(f"{i + 1}/{len(teams_today)}  Updating {slug}")
        updated_rows.extend(scrape_team_gamelog(slug, season, name))
        sleep(sleep_seconds)

    updated_df = pd.DataFrame(updated_rows)
    if updated_df.empty:
        print("No updated rows scraped; leaving file unchanged.")
        return

    # Replace rows for updated teams in this season with fresh scrape
    updated_slugs = set(teams_today["school_slug"].tolist())
    keep_mask = ~(
        (df["season"] == season) & (df["school_slug"].isin(updated_slugs))
    )
    merged = pd.concat([df[keep_mask], updated_df], ignore_index=True)
    merged = clean_gamelogs(merged)

    merged.to_excel(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Update gamelogs for a specific date and append next-game rows."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season year (default: 2026).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the input gamelogs xlsx.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to the output gamelogs xlsx.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / str(args.season)

    input_path = (
        Path(args.input)
        if args.input
        else data_dir / f"NCAAB_{args.season}_Team_Gamelogs_updated.xlsx"
    )
    output_path = (
        Path(args.output)
        if args.output
        else data_dir / f"NCAAB_{args.season}_Team_Gamelogs_updated.xlsx"
    )

    update_gamelogs_for_date(
        input_path=input_path,
        output_path=output_path,
        target_date=args.date,
        season=args.season,
    )


if __name__ == "__main__":
    main()

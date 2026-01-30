import numpy as np
import pandas as pd
from pathlib import Path
from time import sleep

try:
    from .gamelog_scraping import scrape_team_gamelog
except ImportError:
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


def clean_names_gamelogs(df, rename_map=RENAME_MAP):
    df = df.copy()
    df["school_name"] = df["school_name"].str.replace(r"NCAA$", "", regex=True)
    df["opp_name_abbr"] = df["opp_name_abbr"].replace(rename_map)

    valid_schools = set(df["school_name"].unique())
    df = df[df["opp_name_abbr"].isin(valid_schools)]
    return df

def clean_school_names(df):
    df = df.copy()
    df["school_name"] = df["school_name"].str.replace(r"NCAA$", "", regex=True)
    return df


def clean_gamelogs(df, rename_map=RENAME_MAP):
    df = df.copy()
    df["school_name"] = df["school_name"].str.replace(r"NCAA$", "", regex=True)
    df["date"] = pd.to_datetime(df["date"])
    df["opp_name_abbr"] = df["opp_name_abbr"].replace(rename_map)

    valid_schools = set(df["school_name"].unique())
    df = df[df["opp_name_abbr"].isin(valid_schools)]
    return df


def add_features(df):
    df = df.copy()

    # Basic columns
    df["game_location"] = df["game_location"].fillna("")
    df["is_Home"] = df["game_location"].apply(
        lambda x: 1 if x == "" else (0.5 if x == "N" else 0)
    )
    df["score_diff"] = df["team_game_score"] - df["opp_team_game_score"]
    df["win"] = df["team_game_result"].map({"W": 1, "L": 0})

    # Sort + group
    df = df.sort_values(["season", "school_name", "date"])
    g = df.groupby(["season", "school_name"])

    # Ratings (per 100 possessions)
    df["off_rtg"] = 100 * (df["team_game_score"] / df["possessions"])
    df["def_rtg"] = 100 * (df["opp_team_game_score"] / df["possessions"])
    df["net_rtg"] = df["off_rtg"] - df["def_rtg"]

    # Cumulative ratings up to prior game
    cum_pts = g["team_game_score"].transform(lambda s: s.shift(1).cumsum())
    cum_opp_pts = g["opp_team_game_score"].transform(lambda s: s.shift(1).cumsum())
    cum_poss = g["possessions"].transform(lambda s: s.shift(1).cumsum())

    df["cum_off_rtg"] = 100 * (cum_pts / cum_poss)
    df["cum_def_rtg"] = 100 * (cum_opp_pts / cum_poss)
    df["cum_net_rtg"] = df["cum_off_rtg"] - df["cum_def_rtg"]

    df["cum_off_rtg"] = df["cum_off_rtg"].fillna(0)
    df["cum_def_rtg"] = df["cum_def_rtg"].fillna(0)
    df["cum_net_rtg"] = df["cum_net_rtg"].fillna(0)

    # Cumulative ratings home and away
    is_home = df["is_Home"] == 1
    is_away = df["is_Home"] == 0

    home_pts = (
        df.loc[is_home]
        .groupby(["season", "school_name"])["team_game_score"]
        .transform(lambda s: s.shift(1).cumsum())
    )
    home_opp_pts = (
        df.loc[is_home]
        .groupby(["season", "school_name"])["opp_team_game_score"]
        .transform(lambda s: s.shift(1).cumsum())
    )
    home_poss = (
        df.loc[is_home]
        .groupby(["season", "school_name"])["possessions"]
        .transform(lambda s: s.shift(1).cumsum())
    )

    away_pts = (
        df.loc[is_away]
        .groupby(["season", "school_name"])["team_game_score"]
        .transform(lambda s: s.shift(1).cumsum())
    )
    away_opp_pts = (
        df.loc[is_away]
        .groupby(["season", "school_name"])["opp_team_game_score"]
        .transform(lambda s: s.shift(1).cumsum())
    )
    away_poss = (
        df.loc[is_away]
        .groupby(["season", "school_name"])["possessions"]
        .transform(lambda s: s.shift(1).cumsum())
    )

    df["home_cum_net_rtg"] = 0.0
    df["away_cum_net_rtg"] = 0.0

    df.loc[is_home, "home_cum_net_rtg"] = 100 * (home_pts / home_poss) - 100 * (
        home_opp_pts / home_poss
    )
    df.loc[is_away, "away_cum_net_rtg"] = 100 * (away_pts / away_poss) - 100 * (
        away_opp_pts / away_poss
    )

    df[["home_cum_net_rtg", "away_cum_net_rtg"]] = df[
        ["home_cum_net_rtg", "away_cum_net_rtg"]
    ].fillna(0)

    df["home_road_split"] = df["home_cum_net_rtg"] - df["away_cum_net_rtg"]

    # Rolling helper
    def roll_mean(col, window):
        return g[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    # Win pct
    df["win_pct_last_10"] = roll_mean("win", 10).fillna(0)

    # Weighted eFG%
    fg_roll_5 = g["fg"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    fg3_roll_5 = g["fg3"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    fga_roll_5 = g["fga"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    df["efg_pct_last_5"] = (fg_roll_5 + 0.5 * fg3_roll_5) / fga_roll_5

    df["efg_pct_last_5"] = df["efg_pct_last_5"].fillna(0)

    fg_roll_10 = g["fg"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )
    fg3_roll_10 = g["fg3"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )
    fga_roll_10 = g["fga"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum()
    )
    df["efg_pct_last_10"] = (fg_roll_10 + 0.5 * fg3_roll_10) / fga_roll_10

    df["efg_pct_last_10"] = df["efg_pct_last_10"].fillna(0)

    # Rolling stats
    roll_cols = [
        "fta",
        "ast",
        "trb",
        "orb",
        "tov",
        "team_game_score",
        "opp_team_game_score",
        "score_diff",
        "possessions",
    ]
    for col in roll_cols:
        df[f"avg_{col}_last_5"] = roll_mean(col, 5)
        df[f"avg_{col}_last_10"] = roll_mean(col, 10)

    # Fill NaNs
    fill_cols = (
        [f"avg_{c}_last_5" for c in roll_cols]
        + [f"avg_{c}_last_10" for c in roll_cols]
    )
    df[fill_cols] = df[fill_cols].fillna(0)

    # Rest days
    df["prev_game_date"] = g["date"].shift(1)
    df["rest_days"] = (df["date"] - df["prev_game_date"]).dt.days
    df["rest_days"] = df["rest_days"].fillna(7)

    return df


def add_opponent_features(all_df):
    df = all_df.copy()

    opp_cols = [
        "opp_name_abbr",
        "date",
        "opp_team_game_score",
        "rest_days",
        "win_pct_last_10",
        "efg_pct_last_5",
        "efg_pct_last_10",
        "avg_fta_last_5",
        "avg_fta_last_10",
        "avg_ast_last_5",
        "avg_ast_last_10",
        "avg_trb_last_5",
        "avg_trb_last_10",
        "avg_orb_last_5",
        "avg_orb_last_10",
        "avg_tov_last_5",
        "avg_tov_last_10",
        "avg_team_game_score_last_5",
        "avg_team_game_score_last_10",
        "avg_opp_team_game_score_last_5",
        "avg_opp_team_game_score_last_10",
        "avg_score_diff_last_5",
        "avg_score_diff_last_10",
        "avg_possessions_last_5",
        "avg_possessions_last_10",
        "cum_off_rtg",
        "cum_def_rtg",
        "cum_net_rtg",
        "home_cum_net_rtg",
        "away_cum_net_rtg",
        "home_road_split",
    ]

    opp_df = df[opp_cols].add_prefix("opp_")

    merged = pd.merge(
        df,
        opp_df,
        left_on=["date", "school_name", "team_game_score"],
        right_on=["opp_date", "opp_opp_name_abbr", "opp_opp_team_game_score"],
        how="left",
    )

    merged["avg_score_comp_last_10"] = (
        merged["avg_team_game_score_last_10"]
        - merged["opp_avg_team_game_score_last_10"]
    )
    merged["efg_comp_last_10"] = (
        merged["efg_pct_last_10"] - merged["opp_efg_pct_last_10"]
    )
    merged["avg_tov_comp_last_10"] = (
        merged["avg_tov_last_10"] - merged["opp_avg_tov_last_10"]
    )
    merged["avg_orb_comp_last_10"] = (
        merged["avg_orb_last_10"] - merged["opp_avg_orb_last_10"]
    )
    merged["avg_fta_comp_last_10"] = (
        merged["avg_fta_last_10"] - merged["opp_avg_fta_last_10"]
    )
    merged["rest_days_comp"] = merged["rest_days"] - merged["opp_rest_days"]
    merged["net_rtg_comp"] = merged["cum_net_rtg"] - merged["opp_cum_net_rtg"]
    merged["home_road_split_comp"] = merged["home_road_split"] - merged["opp_home_road_split"]
    merged["pace_mismatch_signed"] = merged["avg_possessions_last_10"] - merged["opp_avg_possessions_last_10"]
    merged["net_rtg_home_interaction"] = merged["net_rtg_comp"] * merged["is_Home"]
    

    merged = merged.dropna(subset=["avg_score_comp_last_10"])

    return merged


def calculate_possessions(df):
    """
    Calculate possessions using the accurate formula.

    Formula: 0.5 * (Team_Poss + Opp_Poss)

    Where:
    Poss = FGA + 0.4*FTA - 1.07*(ORB/(ORB+Opp_DRB))*(FGA-FG) + TOV
    """

    # Team possessions
    team_orb_pct = df["orb"] / (df["orb"] + df["opp_drb"])
    team_missed_fg = df["fga"] - df["fg"]
    team_orb_adjustment = 1.07 * team_orb_pct * team_missed_fg

    team_poss = df["fga"] + 0.4 * df["fta"] - team_orb_adjustment + df["tov"]

    # Opponent possessions
    opp_orb_pct = df["opp_orb"] / (df["opp_orb"] + df["drb"])
    opp_missed_fg = df["opp_fga"] - df["opp_fg"]
    opp_orb_adjustment = 1.07 * opp_orb_pct * opp_missed_fg

    opp_poss = (
        df["opp_fga"] + 0.4 * df["opp_fta"] - opp_orb_adjustment + df["opp_tov"]
    )

    # Average of both
    df["possessions"] = 0.5 * (team_poss + opp_poss)
    df["team_possessions"] = team_poss
    df["opp_possessions"] = opp_poss

    return df


def _resolve_base_dir():
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    if base_dir.name == "Daily Scripts" or base_dir.name == "Daily_Scripts":
        base_dir = base_dir.parent
    if base_dir.name == "data":
        base_dir = base_dir.parent
    if not (base_dir / "data").exists() and (base_dir / "NCAAB_Sports_Reference_Scraper").exists():
        base_dir = base_dir / "NCAAB_Sports_Reference_Scraper"
    return base_dir


def create_features(only_season=None, base_dir=None):
    base_dir = Path(base_dir) if base_dir is not None else _resolve_base_dir()
    data_dir = base_dir / "data"

    if only_season == 2026:
        all_df = pd.read_excel(
            data_dir / "2026" / "NCAAB_2026_Team_Gamelogs_now.xlsx", index_col=False
        )
    else:
        df_2023 = pd.read_csv(data_dir / "2023" / "gamelogs_2023.csv", index_col=False)
        df_2024 = pd.read_csv(data_dir / "2024" / "gamelogs_2024.csv", index_col=False)
        df_2025 = pd.read_csv(data_dir / "2025" / "gamelogs_2025.csv", index_col=False)
        df_2026 = pd.read_excel(
            data_dir / "2026" / "NCAAB_2026_Team_Gamelogs_now.xlsx",
            index_col=False,
        )
        all_df = pd.concat([df_2023, df_2024, df_2025, df_2026], ignore_index=True)

    clean_df = clean_gamelogs(all_df)
    merged_df = calculate_possessions(clean_df)
    added_df = add_features(merged_df)
    merged_df = add_opponent_features(added_df)

    output_path = (
        data_dir / "2026" / "features_2026.csv"
        if only_season == 2026
        else data_dir / "merged_dataset.csv"
    )
    merged_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


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


def update_gamelogs_by_date(
    target_date, season=2026, input_path=None, output_path=None
):
    base_dir = _resolve_base_dir()
    data_dir = base_dir / "data" / str(season)

    input_path = (
        Path(input_path)
        if input_path
        else data_dir / f"NCAAB_{season}_Team_Gamelogs_now.xlsx"
    )
    output_path = (
        Path(output_path)
        if output_path
        else data_dir / f"NCAAB_{season}_Team_Gamelogs_now.xlsx"
    )

    update_gamelogs_for_date(
        input_path=input_path,
        output_path=output_path,
        target_date=target_date,
        season=season,
    )

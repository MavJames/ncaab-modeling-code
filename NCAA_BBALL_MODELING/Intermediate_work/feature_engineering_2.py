import numpy as np
import pandas as pd
from pathlib import Path


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
    cum_pts = g["team_game_score"].cumsum().shift(1)
    cum_opp_pts = g["opp_team_game_score"].cumsum().shift(1)
    cum_poss = g["possessions"].cumsum().shift(1)

    df["cum_off_rtg"] = 100 * (cum_pts / cum_poss)
    df["cum_def_rtg"] = 100 * (cum_opp_pts / cum_poss)
    df["cum_net_rtg"] = df["cum_off_rtg"] - df["cum_def_rtg"]

    df["cum_off_rtg"] = df["cum_off_rtg"].fillna(0)
    df["cum_def_rtg"] = df["cum_def_rtg"].fillna(0)
    df["cum_net_rtg"] = df["cum_net_rtg"].fillna(0)

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
        merged["avg_team_game_score_last_10"] - merged["opp_avg_team_game_score_last_10"]
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
    team_orb_pct = df['orb'] / (df['orb'] + df['opp_drb'])
    team_missed_fg = df['fga'] - df['fg']
    team_orb_adjustment = 1.07 * team_orb_pct * team_missed_fg
    
    team_poss = (
        df['fga'] + 
        0.4 * df['fta'] - 
        team_orb_adjustment + 
        df['tov']
    )
    
    # Opponent possessions
    opp_orb_pct = df['opp_orb'] / (df['opp_orb'] + df['drb'])
    opp_missed_fg = df['opp_fga'] - df['opp_fg']
    opp_orb_adjustment = 1.07 * opp_orb_pct * opp_missed_fg
    
    opp_poss = (
        df['opp_fga'] + 
        0.4 * df['opp_fta'] - 
        opp_orb_adjustment + 
        df['opp_tov']
    )
    
    # Average of both
    df['possessions'] = 0.5 * (team_poss + opp_poss)
    df['team_possessions'] = team_poss
    df['opp_possessions'] = opp_poss
    
    return df


def _resolve_base_dir():
    try:
        base_dir = Path(__file__).resolve().parents[1]
    except NameError:
        base_dir = Path.cwd()

    if base_dir.name == "data":
        return base_dir.parent
    return base_dir


def main(only_season=None, base_dir=None):
    base_dir = Path(base_dir) if base_dir is not None else _resolve_base_dir()
    data_dir = base_dir / "data"

    if only_season == 2026:
        all_df = pd.read_excel(
            data_dir / "2026" / "NCAAB_2026_Team_Gamelogs_updated.xlsx", index_col=False
        )
    else:
        df_2023 = pd.read_csv(data_dir / "2023" / "gamelogs_2023.csv", index_col=False)
        df_2024 = pd.read_csv(data_dir / "2024" / "gamelogs_2024.csv", index_col=False)
        df_2025 = pd.read_csv(data_dir / "2025" / "gamelogs_2025.csv", index_col=False)
        df_2026 = pd.read_excel(
            data_dir / "2026" / "NCAAB_2026_Team_Gamelogs_updated.xlsx", index_col=False
        )
        all_df = pd.concat([df_2023, df_2024, df_2025, df_2026], ignore_index=True)

    clean_df = clean_gamelogs(all_df)
    added_df = add_features(clean_df)
    merged_df = add_opponent_features(added_df)
    merged_df = calculate_possessions(merged_df)

    output_path = (
        data_dir / "2026" / "features_2026_new.csv"
        if only_season == 2026
        else data_dir / "merged_dataset.csv"
    )
    merged_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

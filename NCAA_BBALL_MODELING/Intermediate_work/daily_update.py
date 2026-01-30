import pandas as pd

from gamelog_scraping import scrape_all_gamelogs
from data.feature_engineering_2 import main as build_features


def run_daily_update(season=2026):
    df = scrape_all_gamelogs(season)
    output_path = (
        "/Users/mavinjames/Projects/Basketball_Modeling/"
        "NCAAB_Sports_Reference_Scraper/data/"
        f"{season}/NCAAB_{season}_Team_Gamelogs.xlsx"
    )
    df.to_excel(output_path, index=False)
    print(f"Saved gamelogs to {output_path}")

    build_features(only_season=season)


if __name__ == "__main__":
    run_daily_update()

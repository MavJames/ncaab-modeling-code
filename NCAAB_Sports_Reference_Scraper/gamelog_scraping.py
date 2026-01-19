#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError
import pandas as pd
from time import sleep

# GET SCHOOLS
def get_schools(year):
    url = "https://www.sports-reference.com/cbb/seasons/" + str(year) + "-school-stats.html"
    page = urlopen(url).read()
    soup = BeautifulSoup(page, features="lxml")

    table = soup.find("tbody")
    school_dict = {}

    for row in table.find_all("td", {"data-stat": "school_name"}):
        school_name = row.get_text(strip=True)
        a = row.find("a", href=True)
        if not a:
            continue

        link = a["href"].strip()
        school_slug = link.split("/")[3]
        school_dict[school_slug] = school_name

    return school_dict


 
# Function to prevent rate limiting
def fetch_page_safe(url, school):
    max_retries = 3
    retry_count = 0
    page = None

    while retry_count < max_retries and page is None:
        try:
            page = urlopen(url).read()
        except HTTPError as e:
            if e.code == 429:
                retry_count += 1
                wait_time = 10 * retry_count  # 10s, 20s, 30s
                print(
                    f"⏳ Rate limited for {school}. "
                    f"Waiting {wait_time}s (retry {retry_count}/{max_retries})..."
                )
                sleep(wait_time)
            else:
                raise

    return page


# --------------------------------------------------
# SCRAPE ONE TEAM GAMELOG
# --------------------------------------------------
def scrape_team_gamelog(team_slug, year, school_name):
    url = (
        "https://www.sports-reference.com/cbb/schools/"
        + team_slug + "/" + str(year) + "-gamelogs.html"
    )

    print(f"📍 Fetching: {url}")

    page = fetch_page_safe(url, team_slug)
    if page is None:
        print(f"Skipping {team_slug} after retries")
        return []

    soup = BeautifulSoup(page, "lxml")
    table = soup.find("table", id="team_game_log")

    if table is None:
        print(f"❌ No gamelog table for {team_slug}")
        return []

    rows = table.find("tbody").find_all("tr")
    games = []

    for row in rows:
        # Skip future games
        result_cell = row.find("td", {"data-stat": "team_game_result"})
        if result_cell is None or result_cell.get_text(strip=True) == "":
            continue

        game = {
            "school_name": school_name,
            "school_slug": team_slug,
            "season": year
        }

        for cell in row.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat:
                game[stat] = cell.get_text(strip=True)

        games.append(game)

    return games


# --------------------------------------------------
# SCRAPE ALL TEAMS
# --------------------------------------------------
def scrape_all_gamelogs(year):
    schools = get_schools(year)
    schools = dict(list(schools.items())[175:375])
    all_games = []

    for i, (slug, name) in enumerate(schools.items(), 1):
        print(f"{i}/{len(schools)}  Scraping {slug}")

        team_games = scrape_team_gamelog(slug, year, name)
        all_games.extend(team_games)

        # 👇 SAME SAFE DELAY YOU USED BEFORE
        sleep(5)

    return pd.DataFrame(all_games)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    SEASON = 2025

    df = scrape_all_gamelogs(SEASON)

    output_path = (
        "/Users/mavinjames/Desktop/"
        + "NCAAB_" + str(SEASON) + "_Team_Gamelogs2.xlsx"
    )

    df.to_excel(output_path, index=False)

    print(f"\n✅ DONE — saved {len(df)} rows to:")
    print(output_path)

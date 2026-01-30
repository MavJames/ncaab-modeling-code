#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError
from http.client import IncompleteRead
import pandas as pd
from time import sleep
from pathlib import Path


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
        except IncompleteRead:
            retry_count += 1
            wait_time = 5 * retry_count
            print(
                f"⚠️ Incomplete read for {school}. "
                f"Retrying in {wait_time}s (retry {retry_count}/{max_retries})..."
            )
            sleep(wait_time)

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

    parsed_rows = []
    for row in rows:
        result_cell = row.find("td", {"data-stat": "team_game_result"})
        if result_cell is None:
            continue
        parsed_rows.append((row, result_cell.get_text(strip=True)))

    last_completed_idx = None
    for idx, (_, result) in enumerate(parsed_rows):
        if result != "":
            last_completed_idx = idx

    future_added = False
    for idx, (row, result) in enumerate(parsed_rows):
        if result != "":
            game = {
                "school_name": school_name,
                "school_slug": team_slug,
                "season": year,
            }
            for cell in row.find_all(["th", "td"]):
                stat = cell.get("data-stat")
                if stat:
                    game[stat] = cell.get_text(strip=True)
            games.append(game)
            continue

        if last_completed_idx is None:
            is_after_last = True
        else:
            is_after_last = idx > last_completed_idx

        if is_after_last and not future_added:
            game = {
                "school_name": school_name,
                "school_slug": team_slug,
                "season": year,
            }
            for cell in row.find_all(["th", "td"]):
                stat = cell.get("data-stat")
                if stat:
                    game[stat] = cell.get_text(strip=True)
            games.append(game)
            future_added = True
    return games


# --------------------------------------------------
# SCRAPE ALL TEAMS
# --------------------------------------------------
def scrape_all_gamelogs(year, limit=5):
    schools = get_schools(year)
    if limit is not None:
        schools = dict(list(schools.items())[:limit])
    all_games = []

    for i, (slug, name) in enumerate(schools.items(), 1):
        print(f"{i}/{len(schools)}  Scraping {slug}")

        team_games = scrape_team_gamelog(slug, year, name)
        all_games.extend(team_games)

        # 👇 SAME SAFE DELAY YOU USED BEFORE
        sleep(8)

    return pd.DataFrame(all_games)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    SEASON = 2026

    df = scrape_all_gamelogs(SEASON)

    base_dir = Path(__file__).resolve().parent
    output_path = (
    base_dir / "data" / "2026" / f"NCAAB_{SEASON}_Team_Gamelogs_now.xlsx"
)
    df.to_excel(output_path, index=False)

    print(f"\n✅ DONE — saved {len(df)} rows to:")
    print(output_path)

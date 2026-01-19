#!/usr/bin/env python
# coding: utf-8

# In[16]:


from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError
import pandas as pd
from time import sleep
import sys


# In[17]:


def get_schools(year):
    url = "https://www.sports-reference.com/cbb/seasons/" + str(year) + "-school-stats.html"
    page = urlopen(url).read()
    soup = BeautifulSoup(page, features="lxml")
    count  = 0
    table = soup.find("tbody")
    school_dict = dict()
    for row in table.findAll('td', {"data-stat": "school_name"}):
        school_name = row.getText()
        for a in row.find_all('a', href=True):
            link = a['href'].strip()
            name = link[13:].split("/")[0]
            school_dict[name] = school_name
            
    return school_dict


# In[18]:


def remove_substring_from_dict_values(dictionary, substring):
    for key, value in dictionary.items():
        if type(value) == str:
            dictionary[key] = value.replace(substring, "")
    return dictionary


# In[19]:


def scrape_season(school_set, year):
    dfs = []
    final_df=pd.DataFrame()
    for school in school_set:
        print(school + '-' + str(year))
        url = "https://www.sports-reference.com/cbb/schools/" + school + "/" + str(year) + "-schedule.html"

        print(f"📍 Fetching URL: {url}")
        
        # Add retry logic for rate limiting
        max_retries = 3
        retry_count = 0
        page = None
        
        while retry_count < max_retries and page is None:
            try:
                page = urlopen(url).read()
            except HTTPError as e:
                if e.code == 429:
                    retry_count += 1
                    wait_time = 10 * retry_count  # Exponential backoff: 10s, 20s, 30s
                    print(f"Rate limited. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                    sleep(wait_time)
                else:
                    raise
        
        if page is None:
            print(f"Skipping {school} after {max_retries} retries")
            continue
            
        soup = BeautifulSoup(page, features="lxml")
        pre_df = dict()
        if (soup.find_all("tbody")[1] != None):
            table = soup.find_all("tbody")[1]
            featuresWanted =  {'opp_name', 'pts', 'opp_pts', 'game_location','game_result','overtimes','wins','losses', 'date_game'} #add more features here!!

            rows = table.find_all('tr')  # Updated from findChildren
            for row in rows:
                if (row.find('th', {"scope":"row"}) != None):

                    for f in featuresWanted:
                        cell = row.find("td",{"data-stat": f})

                        a = cell.text.strip().encode()
                        text=a.decode("utf-8")
                        if f in pre_df:
                            pre_df[f].append(text)
                        else:
                            pre_df[f]=[text]
            df = pd.DataFrame.from_dict(pre_df)
            df["opp_name"]= df["opp_name"].apply(lambda row: (row.split("(")[0]).rstrip())
            df["school_name"]=school_set[school]
            df["school_name"] = df["school_name"].apply(lambda row: (row.split("(")[0]).rstrip())
            final_df=pd.concat([final_df,df])
            final_df = final_df[['date_game', 'school_name','game_location','opp_name','pts','opp_pts','overtimes','game_result','wins','losses']]
            final_df['game_location'].replace(['@'], 'Away')
            final_df['game_location'].replace(['N'], 'Neutral')
            final_df['game_location'].replace([''], 'Home')
            final_df['MOV'] = pd.to_numeric(final_df['pts']) - pd.to_numeric(final_df['opp_pts'])
            sleep(5)  # Increased sleep time from 2 to 5 seconds
    return final_df


# In[20]:


current_year = 2023
for year in range(current_year, current_year - 5, -1):
    try:
        print(f"\n=== Starting year {year} ===")
        schools = get_schools(year)
        schools_cleaned = remove_substring_from_dict_values(schools, "\xa0NCAA")
        season_data = scrape_season(schools_cleaned, year)
        # Fixed deprecated append method
        season_data = pd.concat([season_data, season_data])
        # Updated file path to your desktop
        season_data.to_excel("/Users/mavinjames/Desktop/NCAAB_" + str(year) + "_Boxscores.xlsx", index = False)
        print(f"=== Completed year {year} ===\n")
    except Exception as err:
        print(f"Error processing year {year}: {err}")
        continue

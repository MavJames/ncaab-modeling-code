# NCAA Basketball Data Collection Guide for Modeling

## The Problem
Web scraping Sports Reference is difficult due to:
- Rate limiting (getting blocked)
- Sportsipy library compatibility issues
- Time-consuming manual processes

## The Professional Solution: Use Existing Datasets

Real data analysts typically use pre-compiled datasets rather than scraping. Here are your best options:

---

## OPTION 1: Kaggle Datasets (RECOMMENDED)

### Available Datasets:

1. **College Basketball Dataset by Andrew Sundberg**
   - URL: https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset
   - Contains: Team stats, game-by-game results, multiple seasons
   - Good for: Building predictive models

2. **NCAA Basketball Dataset (Official)**
   - URL: https://www.kaggle.com/datasets/ncaa/ncaa-basketball
   - Contains: Comprehensive game data
   - Multiple seasons available

3. **March Madness Historical Dataset (2002-2025)**
   - URL: https://www.kaggle.com/datasets/jonathanpilafas/2024-march-madness-statistical-analysis
   - Contains: 20+ years of tournament data
   - Good for: Historical analysis and modeling

4. **College Basketball Play-by-Play 2023-24**
   - URL: https://www.kaggle.com/datasets/robbypeery/college-basketball-pbp-23-24
   - Contains: Detailed play-by-play data
   - Good for: Advanced analytics

### How to Use Kaggle Datasets:

1. **Create a free Kaggle account** (if you don't have one)
   - Go to: https://www.kaggle.com
   - Sign up with Google/email

2. **Download the dataset**
   - Visit the dataset URL above
   - Click "Download" button
   - Extract the ZIP file

3. **Load into Python**
   ```python
   import pandas as pd
   
   # Load the CSV file
   df = pd.read_csv('/path/to/downloaded/dataset.csv')
   
   # Start analyzing!
   print(df.head())
   print(df.columns)
   ```

---

## OPTION 2: Use Kaggle API (Automated)

Install Kaggle CLI:
```bash
pip install kaggle
```

Setup authentication:
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token" (downloads kaggle.json)
4. Move file to: `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Download dataset via command line:
```bash
# Example: Download Andrew Sundberg's dataset
kaggle datasets download -d andrewsundberg/college-basketball-dataset

# Unzip
unzip college-basketball-dataset.zip
```

---

## OPTION 3: BigQuery Public Dataset (Google Cloud)

Google has NCAA basketball data in BigQuery:
- Dataset: `bigquery-public-data.ncaa_basketball`
- Tables include: games, teams, players, etc.
- Requires: Free Google Cloud account (includes $300 credit)

Access via Python:
```python
from google.cloud import bigquery

client = bigquery.Client()

query = """
SELECT *
FROM `bigquery-public-data.ncaa_basketball.mbb_games_sr`
WHERE season >= 2020
LIMIT 1000
"""

df = client.query(query).to_dataframe()
```

---

## OPTION 4: Ken Pomeroy's Data (kenpom.com)

- Premium service ($20/year)
- Industry standard for advanced metrics
- Tempo-free statistics
- Efficiency ratings
- Used by many professional analysts

---

## RECOMMENDED WORKFLOW FOR YOUR PROJECT:

### Step 1: Get the Data
Use **Kaggle** - it's free, comprehensive, and widely used in the industry.

Start with: **Andrew Sundberg's College Basketball Dataset**
https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset

### Step 2: Explore the Data
```python
import pandas as pd

df = pd.read_csv('cbb.csv')

# See what you have
print(df.info())
print(df.describe())
print(df.head(20))
```

### Step 3: Build Your Model
With clean data, you can focus on:
- Feature engineering
- Model selection
- Training/testing
- Evaluation

### Step 4: Document Your Process
This is what impresses employers:
- Data source citation
- Data cleaning steps
- Model rationale
- Performance metrics
- Code documentation

---

## Why This Approach is Better:

✅ **No rate limiting** - Data is already compiled
✅ **Clean and structured** - Ready for analysis
✅ **Multiple seasons** - Better for modeling
✅ **Industry standard** - This is what professionals do
✅ **Reproducible** - Others can verify your work
✅ **Time-efficient** - Start modeling immediately

---

## Next Steps:

1. Go to Kaggle and download a dataset
2. Load it into Python/Pandas
3. Start exploring and building your model
4. Document everything for your portfolio

This approach shows you understand:
- Data sourcing best practices
- Efficient workflows
- Industry standards
- Time management

Much better for a job interview than explaining why you spent weeks fighting with web scrapers!

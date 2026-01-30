# NCAAB Basketball Score Prediction Model

A machine learning project to predict score differentials in NCAA Division I Men's Basketball games, backed by a Streamlit web application for interactive predictions.

## Project Overview

This project scrapes historical game log data from Sports Reference, builds a predictive model for score differentials, and provides an interactive web interface where users can input two teams and receive a predicted score differential.

## Project Structure

```
Basketball_Modeling/
├── NCAAB_Sports_Reference_Scraper/    # Data collection scripts
│   ├── data/                           # Raw scraped data (not tracked in git)
│   └── scraper.py                      # Sports Reference scraper
├── data/                               # Processed datasets (not tracked in git)
│   ├── raw/                            # Raw scraped data
│   ├── processed/                      # Cleaned and feature-engineered data
│   └── training/                       # Train/test splits
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── src/                                # Source code
│   ├── data/
│   │   ├── scraper.py                  # Data collection functions
│   │   └── preprocessing.py            # Data cleaning and feature engineering
│   ├── models/
│   │   ├── train.py                    # Model training scripts
│   │   └── predict.py                  # Prediction functions
│   └── utils/
│       └── helpers.py                  # Utility functions
├── models/                             # Saved trained models
│   └── score_differential_model.pkl
├── app/                                # Streamlit application
│   ├── streamlit_app.py                # Main Streamlit app
│   └── components/                     # UI components
├── tests/                              # Unit tests
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Features

### Data Collection
- Scrapes NCAAB game logs from Sports Reference for multiple seasons
- Collects box score statistics including:
  - Team statistics (points, rebounds, assists, turnovers, etc.)
  - Shooting percentages (FG%, 3P%, FT%)
  - Advanced metrics

### Model Development
- **Target Variable**: Score differential (Team A - Team B)
- **Features**: 
  - Team statistics (offensive/defensive ratings)
  - Recent form (rolling averages)
  - Head-to-head history
  - Home/away status
  - Season-to-date performance metrics
- **Models to Explore**:
  - Linear Regression (baseline)
  - Random Forest
  - Gradient Boosting (XGBoost/LightGBM)
  - Neural Networks

### Streamlit App
- Interactive web interface for predictions
- Input: Two team names
- Output: Predicted score differential with confidence intervals
- Visualization of key factors influencing the prediction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Basketball_Modeling.git
cd Basketball_Modeling
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Validate Your Data
Before processing, check that your files are in the correct location:
```bash
python src/data/validate_data.py
python src/data/validate_data.py --quality-check  # For detailed analysis
```

### 2. Combine Data Files
Process a single season:
```bash
python src/data/combine_data.py --season 2023
```

Or process all seasons at once:
```bash
python src/data/process_all_seasons.py
```

This creates:
- `data/{season}/gamelogs_{season}.csv` - Combined game logs
- `data/{season}/full_stats_{season}.csv` - Complete dataset with team stats

### 3. Data Preprocessing (Coming Soon)
```bash
python src/data/preprocessing.py
```

### 4. Model Training (Coming Soon)
```bash
python src/models/train.py --model xgboost --output models/score_differential_model.pkl
```

### 5. Run Streamlit App (Coming Soon)
```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Requirements

Key dependencies (see `requirements.txt` for full list):
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- beautifulsoup4
- requests
- matplotlib
- seaborn

## Development Roadmap

- [x] Build Sports Reference scraper
- [x] Create data combination scripts
- [x] Add data validation tools
- [ ] Data preprocessing and feature engineering
- [ ] Exploratory data analysis
- [ ] Baseline model development
- [ ] Advanced model development and tuning
- [ ] Model evaluation and validation
- [ ] Build Streamlit application
- [ ] Deploy application
- [ ] Add real-time data updates

## Model Performance

*To be updated after model training*

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Baseline | - | - | - |
| Random Forest | - | - | - |
| XGBoost | - | - | - |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sourced from [Sports Reference](https://www.sports-reference.com/)
- Inspired by sports analytics community

## Contact

For questions or feedback, please open an issue or contact [your email/contact info].

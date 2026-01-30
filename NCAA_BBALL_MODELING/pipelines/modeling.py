from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "ft_pct",
    "opp_ft_pct",
    "fg_pct",
    "fg3a",
    "fg3_pct",
    "fg",
    "fga",
    "fg2a",
    "fg3",
    "ft",
    "fta",
    "efg_pct",
    "fg2_pct",
    "orb",
    "drb",
    "opp_fg3",
    "opp_fg_pct",
    "blk",
    "opp_fg",
    "pf",
    "tov",
    "ast",
    "stl",
    "trb",
    "team_game_result",
    "opp_efg_pct",
    "opp_ft",
    "opp_fta",
    "opp_orb",
    "opp_drb",
    "opp_trb",
    "opp_ast",
    "opp_stl",
    "opp_blk",
    "opp_tov",
    "opp_fg3a",
    "opp_fg3_pct",
    "opp_fg2",
    "opp_fg2a",
    "opp_fg2_pct",
    "opp_fga",
    "team_possessions",
    "opp_possessions",
    "opp_pf",
    "possessions",
    "off_rtg",
    "def_rtg",
    "win",
    "fg2",
    "net_rtg",
    "team_game_num_season",
    "opp_opp_team_game_score",
    "team_game_score",
    "score_diff",
    "opp_team_game_score",
]


DEFAULT_BASELINE_FEATURES = [
    "net_rtg_comp",
    "is_Home",
    "net_rtg_home_interaction",
]


DEFAULT_RESIDUAL_FEATURES = [
    "avg_score_comp_last_10",
    "efg_comp_last_10",
    "avg_tov_comp_last_10",
    "pace_mismatch_signed",
    "avg_orb_comp_last_10",
    "avg_fta_comp_last_10",
    "rest_days_comp",
    "home_road_split_comp",
]


@dataclass
class ModelBundle:
    baseline_model: object
    residual_model: object
    baseline_features: list[str]
    residual_features: list[str]


@dataclass
class TrainResults:
    model_bundle: ModelBundle
    test_eval: pd.DataFrame


try:
    from NCAA_BBALL_MODELING.utils import _resolve_base_dir
except ImportError:
    from utils import _resolve_base_dir


def load_training_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load merged dataset and drop rows missing required columns."""
    if path is None:
        base_dir = _resolve_base_dir()
        path = base_dir / "data" / "merged_dataset.csv"

    df = pd.read_csv(path)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_rtg_home_interaction"] = df["net_rtg_comp"] * df["is_Home"]
    return df


def train_models(
    df: pd.DataFrame,
    *,
    seasons_train: Iterable[int],
    season_test: int,
    hca_alpha: float = 5.0,
    residual_cap: float = 20.0,
    baseline_features: Optional[list[str]] = None,
    residual_features: Optional[list[str]] = None,
) -> TrainResults:
    """Train baseline and residual models on historical seasons."""
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    baseline_features = baseline_features or DEFAULT_BASELINE_FEATURES
    residual_features = residual_features or DEFAULT_RESIDUAL_FEATURES

    df = add_interactions(df)

    train_df = df[df["season"].isin(seasons_train)].copy()
    test_df = df[df["season"] == season_test].copy()

    # Baseline model
    Xb_train = train_df[baseline_features]
    yb_train = train_df["score_diff"].astype(float)
    mask_b = Xb_train.notna().all(axis=1) & yb_train.notna()
    Xb_train, yb_train = Xb_train[mask_b], yb_train[mask_b]

    baseline_model = Ridge(alpha=hca_alpha)
    baseline_model.fit(Xb_train, yb_train)

    train_df["expected_margin"] = baseline_model.predict(train_df[baseline_features])
    test_df["expected_margin"] = baseline_model.predict(test_df[baseline_features])

    # Residual model
    train_df["residual_margin"] = train_df["score_diff"] - train_df["expected_margin"]
    train_df["residual_margin_capped"] = train_df["residual_margin"].clip(
        -residual_cap, residual_cap
    )

    Xr_train = train_df[residual_features]
    yr_train = train_df["residual_margin_capped"].astype(float)
    mask_r = Xr_train.notna().all(axis=1) & yr_train.notna()
    Xr_train, yr_train = Xr_train[mask_r], yr_train[mask_r]

    residual_model = XGBRegressor(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=20,
        reg_alpha=1.0,
        reg_lambda=3.0,
        objective="reg:pseudohubererror",
        random_state=42,
    )
    residual_model.fit(Xr_train, yr_train)

    # Test predictions
    Xr_test = test_df[residual_features]
    y_test = test_df["score_diff"].astype(float)
    mask_test = Xr_test.notna().all(axis=1) & y_test.notna()
    Xr_test, y_test = Xr_test[mask_test], y_test[mask_test]
    expected_test = test_df.loc[mask_test, "expected_margin"]

    pred_residual = residual_model.predict(Xr_test)
    pred_final = expected_test.values + pred_residual

    test_eval = test_df.loc[mask_test].copy()
    test_eval["pred_final"] = pred_final

    bundle = ModelBundle(
        baseline_model=baseline_model,
        residual_model=residual_model,
        baseline_features=baseline_features,
        residual_features=residual_features,
    )

    return TrainResults(model_bundle=bundle, test_eval=test_eval)


def predict_from_features(
    features_df: pd.DataFrame,
    *,
    model_bundle: ModelBundle,
) -> pd.DataFrame:
    """Predict scores from a feature dataframe."""
    df = add_interactions(features_df)

    df["pred_baseline"] = model_bundle.baseline_model.predict(
        df[model_bundle.baseline_features]
    )
    df["pred_residual"] = model_bundle.residual_model.predict(
        df[model_bundle.residual_features]
    )
    df["pred_final"] = df["pred_baseline"] + df["pred_residual"]
    return df


def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def build_favorites_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    favored_df = pred_df[pred_df["pred_final"] > 0].copy()
    favored_df["team_a"] = favored_df[["school_name", "opp_name_abbr"]].min(axis=1)
    favored_df["team_b"] = favored_df[["school_name", "opp_name_abbr"]].max(axis=1)
    favored_df = favored_df.drop_duplicates(["date", "team_a", "team_b"])

    keep_cols = [
        "date",
        "school_name",
        "opp_name_abbr",
        "pred_final",
        "pred_baseline",
        "pred_residual",
        "is_Home",
        "team_game_score",
        "opp_team_game_score",
        "score_diff",
    ]
    return favored_df[keep_cols]


def save_predictions(pred_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    return output_path


def run_modeling_pipeline(
    *,
    training_path: Optional[str | Path] = None,
    features_path: Optional[str | Path] = None,
    predictions_path: Optional[str | Path] = None,
    seasons_train: Iterable[int] = (2023, 2024, 2025),
    season_test: int = 2026,
    hca_alpha: float = 5.0,
    residual_cap: float = 20.0,
    baseline_features: Optional[list[str]] = None,
    residual_features: Optional[list[str]] = None,
    favorites_only: bool = True,
) -> Path:
    """Train models and write predictions to CSV."""
    df = load_training_data(training_path)
    results = train_models(
        df,
        seasons_train=seasons_train,
        season_test=season_test,
        hca_alpha=hca_alpha,
        residual_cap=residual_cap,
        baseline_features=baseline_features,
        residual_features=residual_features,
    )

    if features_path is None:
        base_dir = _resolve_base_dir()
        features_path = (
            base_dir / "data" / str(season_test) / f"features_{season_test}.csv"
        )

    features_df = load_features(features_path)
    pred_df = predict_from_features(features_df, model_bundle=results.model_bundle)

    if favorites_only:
        pred_df = build_favorites_predictions(pred_df)

    if predictions_path is None:
        base_dir = _resolve_base_dir()
        predictions_path = base_dir / "data" / "predictions.csv"

    return save_predictions(pred_df, predictions_path)

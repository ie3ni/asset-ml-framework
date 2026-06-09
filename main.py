# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler

import config.asset_config as asset_cfg
import config.model_config as model_cfg
import config.feature_config as feature_cfg

from data.downloader import download_data

from features.technical_features import build_features

from models.model_factory import build_model

from evaluation.metrics import calculate_metrics

from reporting.report_generator import generate_report

from reporting.save_results import save_results

from reporting.latest_prediction import (
    get_latest_prediction,
    print_latest_prediction
)
import matplotlib
matplotlib.use("Agg")
# ============================================================
# DOWNLOAD DATA
# ============================================================

df = download_data(
    asset_cfg.TICKER,
    asset_cfg.PERIOD,
    asset_cfg.INTERVAL
)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.columns = [c.lower() for c in df.columns]

print("\nData Head:")
print(df.head())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df = build_features(df, feature_cfg)

# ============================================================
# FEATURES
# ============================================================

features = [
    'return_1d',
    'momentum_fast',
    'momentum_slow',
    'vol_fast',
    'vol_slow',
    'ma_spread',
    'volume_change',
    'rsi'
]

X = df[features]
y = df['target']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

split_index = int(len(df) * model_cfg.TRAIN_SPLIT)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ============================================================
# SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# BUILD MODEL
# ============================================================

model = build_model(model_cfg)

print(f"\nTraining model: {model_cfg.ACTIVE_MODEL}")

# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(X_train_scaled, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(X_test_scaled)

# ============================================================
# METRICS LAYER (NEW)
# ============================================================

metrics = calculate_metrics(
    model=model,
    predictions=predictions,
    y_test=y_test,
    df=df,
    split_index=split_index
)

# ============================================================
# GENERATE REPORT (NOW USES METRICS)
# ============================================================

(
    comparison_df,
    importance_df,
    report_text,
    feature_importance_fig,
    accuracy_fig
) = generate_report(
    metrics=metrics,
    model=model,
    predictions=predictions,
    y_test=y_test,
    features=features,
    df=df,
    split_index=split_index
)

# ============================================================
# LATEST FORECAST
# ============================================================

prediction_text = get_latest_prediction(
    model,
    scaler,
    X,
    df,
    metrics["persistence_accuracy"]
)

print_latest_prediction(prediction_text)

# ============================================================
# CONFIG SNAPSHOT
# ============================================================

config_dict = {
    "ticker": asset_cfg.TICKER,
    "period": asset_cfg.PERIOD,
    "interval": asset_cfg.INTERVAL,
    "train_split": model_cfg.TRAIN_SPLIT,
    "active_model": model_cfg.ACTIVE_MODEL,

    "n_estimators": getattr(model_cfg, "N_ESTIMATORS", None),
    "max_depth": getattr(model_cfg, "MAX_DEPTH", None),
    "min_samples_split": getattr(model_cfg, "MIN_SAMPLES_SPLIT", None)
}

# ============================================================
# SAVE RESULTS
# ============================================================

save_results(
    comparison_df,
    importance_df,
    report_text,
    prediction_text,
    config_dict,
    feature_importance_fig,
    accuracy_fig
)
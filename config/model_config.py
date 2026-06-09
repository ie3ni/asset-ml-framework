# ============================================================
# MODEL SELECTION
# ============================================================

#ACTIVE_MODEL = "random_forest"
#ACTIVE_MODEL = "xgboost"

ACTIVE_MODEL  = "logistic_regression"

# ============================================================
# GENERAL SETTINGS
# ============================================================

TRAIN_SPLIT = 0.80

# ============================================================
# RANDOM FOREST
# ============================================================

RF_N_ESTIMATORS = 300

RF_MAX_DEPTH = 8

RF_MIN_SAMPLES_SPLIT = 10

RF_RANDOM_STATE = 42

# ============================================================
# LOGISTIC REGRESSION
# ============================================================

LR_C = 1.0

LR_MAX_ITER = 1000

LR_PENALTY = "l2"

LR_SOLVER = "lbfgs"

LR_RANDOM_STATE = 42
# ============================================================
# XGBOOST
# ============================================================


XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
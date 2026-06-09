# models/xgboost_model.py

from xgboost import XGBClassifier

def build_xgb(config):

    model = XGBClassifier(
        n_estimators=getattr(config, "N_ESTIMATORS", 200),
        max_depth=getattr(config, "MAX_DEPTH", 5),
        learning_rate=getattr(config, "LEARNING_RATE", 0.1),
        subsample=getattr(config, "SUBSAMPLE", 0.8),
        colsample_bytree=getattr(config, "COLSAMPLE_BYTREE", 0.8),
        random_state=getattr(config, "RANDOM_STATE", 42),
        eval_metric="logloss"
    )
    return model

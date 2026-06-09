from models.random_forest_model import build_rf
from models.logistic_regression_model import build_lr
from models.xgboost_model import build_xgb


def build_model(config):

    if config.ACTIVE_MODEL == "random_forest":
        return build_rf(config)

    elif config.ACTIVE_MODEL == "logistic_regression":
        return build_lr(config)

    elif config.ACTIVE_MODEL == "xgboost":
        return build_xgb(config)

    else:
        raise ValueError(f"Unknown model: {config.ACTIVE_MODEL}")
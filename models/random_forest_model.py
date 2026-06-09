from sklearn.ensemble import RandomForestClassifier


def build_rf(config):

    model = RandomForestClassifier(

        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
        random_state=config.RF_RANDOM_STATE
    )

    return model
from sklearn.linear_model import LogisticRegression

def build_lr(config):

    return LogisticRegression(
        C=config.LR_C,
        max_iter=config.LR_MAX_ITER,
        penalty=config.LR_PENALTY,
        solver=config.LR_SOLVER,
        random_state=config.LR_RANDOM_STATE
    )
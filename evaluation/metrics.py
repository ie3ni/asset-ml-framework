# ============================================================
# IMPORTS
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from evaluation.baselines import (
    always_up,
    persistence
)

# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
        model,
        predictions,
        y_test,
        df,
        split_index):

    # ============================================================
    # MODEL NAME
    # ============================================================

    model_name = (
        type(model).__name__
    )

    # ============================================================
    # MODEL METRICS
    # ============================================================

    model_accuracy = (
        accuracy_score(
            y_test,
            predictions
        )
    )

    report_text = (
        classification_report(
            y_test,
            predictions
        )
    )

    # ============================================================
    # ALWAYS UP BASELINE
    # ============================================================

    always_up_predictions = (
        always_up(y_test)
    )

    always_up_accuracy = (
        accuracy_score(
            y_test,
            always_up_predictions
        )
    )

    # ============================================================
    # PERSISTENCE BASELINE
    # ============================================================

    persistence_predictions = (
        persistence(
            df,
            split_index
        )
    )

    persistence_accuracy = (
        accuracy_score(
            y_test,
            persistence_predictions
        )
    )

    # ============================================================
    # RETURN METRICS
    # ============================================================

    metrics = {

        "model_name":
            model_name,

        "model_accuracy":
            model_accuracy,

        "classification_report":
            report_text,

        "always_up_accuracy":
            always_up_accuracy,

        "persistence_accuracy":
            persistence_accuracy
    }

    return metrics
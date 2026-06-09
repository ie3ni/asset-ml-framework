# ============================================================
# REPORT GENERATOR (METRICS-BASED)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


def generate_report(
        metrics,
        model,
        predictions,
        y_test,
        features,
        df,
        split_index):

    # ============================================================
    # MODEL RESULTS
    # ============================================================

    print("\n==============================")
    print(f"{metrics['model_name']} RESULTS")
    print("==============================")

    print(
        f"\nAccuracy: "
        f"{metrics['model_accuracy']:.4f}"
    )

    print("\nClassification Report:\n")
    print(metrics["classification_report"])

    # ============================================================
    # BASELINES (FROM METRICS)
    # ============================================================

    print("\n==============================")
    print("BASELINE #1")
    print("ALWAYS PREDICT UP")
    print("==============================")

    print(
        f"\nAccuracy: "
        f"{metrics['always_up_accuracy']:.4f}"
    )

    print("\n==============================")
    print("BASELINE #2")
    print("PERSISTENCE MODEL")
    print("==============================")

    print(
        f"\nAccuracy: "
        f"{metrics['persistence_accuracy']:.4f}"
    )

    # ============================================================
    # COMPARISON TABLE
    # ============================================================

    comparison = pd.DataFrame({
        "Model": [
            metrics["model_name"],
            "Always Up",
            "Persistence"
        ],
        "Accuracy": [
            metrics["model_accuracy"],
            metrics["always_up_accuracy"],
            metrics["persistence_accuracy"]
        ]
    })

    comparison = comparison.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\n==============================")
    print("MODEL COMPARISON")
    print("==============================")
    print(comparison)

    # ============================================================
    # FEATURE IMPORTANCE (MODEL-AGNOSTIC)
    # ============================================================

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_
        importance_method = "Tree Feature Importance"

    elif hasattr(model, "coef_"):
        importance_values = abs(model.coef_[0])
        importance_method = "Absolute Coefficients"
    elif hasattr(model, "feature_importances_"):
        importance_method = "Tree-Based Importance (XGBoost / RF)"

    else:
        importance_values = [0] * len(features)
        importance_method = "Unsupported Model"

    importance = pd.DataFrame({
        "feature": features,
        "importance": importance_values
    }).sort_values(by="importance", ascending=False)

    print(f"\nFeature Importance ({importance_method}):\n")
    print(importance)

    # ============================================================
    # PLOT FEATURE IMPORTANCE
    # ============================================================

    feature_importance_fig = plt.figure(figsize=(10, 6))

    plt.bar(
        importance["feature"],
        importance["importance"]
    )

    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.tight_layout()


    # ============================================================
    # PLOT MODEL COMPARISON
    # ============================================================

    accuracy_fig = plt.figure(figsize=(8, 5))

    plt.bar(
        comparison["Model"],
        comparison["Accuracy"]
    )

    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()


    # ============================================================
    # RETURN
    # ============================================================

    return (
        comparison,
        importance,
        metrics["classification_report"],
        feature_importance_fig,
        accuracy_fig
    )

plt.close('all')
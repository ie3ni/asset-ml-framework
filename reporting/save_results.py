# ============================================================
# SAVE RESULTS
# ============================================================

import os
import json

from datetime import datetime

# ============================================================
# CREATE RUN FOLDER
# ============================================================

def create_run_folder(config_dict):

    from datetime import datetime
    import os

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ticker = config_dict.get("ticker", "asset")
    model = config_dict.get("active_model", "model")

    # clean names (important for file safety)
    ticker = ticker.replace("/", "_")
    model = model.replace(" ", "_")

    folder_name = f"{ticker}_{model}_run_{timestamp}"

    run_folder = os.path.join(
        "results",
        folder_name
    )

    os.makedirs(run_folder, exist_ok=True)

    return run_folder

# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

def save_comparison(
        comparison_df,
        run_folder):

    filepath = os.path.join(
        run_folder,
        "model_comparison.csv"
    )

    comparison_df.to_csv(
        filepath,
        index=False
    )

# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
        importance_df,
        run_folder):

    filepath = os.path.join(
        run_folder,
        "feature_importance.csv"
    )

    importance_df.to_csv(
        filepath,
        index=False
    )

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

def save_classification_report(
        report_text,
        run_folder):

    filepath = os.path.join(
        run_folder,
        "classification_report.txt"
    )

    with open(
            filepath,
            "w") as f:

        f.write(
            report_text
        )

# ============================================================
# SAVE LATEST PREDICTION
# ============================================================

def save_latest_prediction(
        prediction_text,
        run_folder):

    filepath = os.path.join(
        run_folder,
        "latest_prediction.txt"
    )

    with open(
            filepath,
            "w") as f:

        f.write(
            prediction_text
        )

# ============================================================
# SAVE CONFIG SNAPSHOT
# ============================================================

def save_config_snapshot(
        config_dict,
        run_folder):

    filepath = os.path.join(
        run_folder,
        "config_snapshot.json"
    )

    with open(
            filepath,
            "w") as f:

        json.dump(

            config_dict,

            f,

            indent=4
        )

# ============================================================
# SAVE FIGURES
# ============================================================

def save_figure(
        figure,
        filename,
        run_folder):

    filepath = os.path.join(
        run_folder,
        filename
    )

    figure.savefig(
        filepath,
        bbox_inches="tight"
    )

# ============================================================
# MASTER SAVE FUNCTION
# ============================================================

def save_results(

        comparison_df,

        importance_df,

        classification_report_text,

        prediction_text,

        config_dict,

        feature_importance_fig,

        accuracy_fig):

    run_folder = (
        create_run_folder(config_dict)
    )

    save_comparison(

        comparison_df,

        run_folder
    )

    save_feature_importance(

        importance_df,

        run_folder
    )

    save_classification_report(

        classification_report_text,

        run_folder
    )

    save_latest_prediction(

        prediction_text,

        run_folder
    )

    save_config_snapshot(

        config_dict,

        run_folder
    )

    save_figure(

        feature_importance_fig,

        "feature_importance.png",

        run_folder
    )

    save_figure(

        accuracy_fig,

        "model_accuracy.png",

        run_folder
    )

    print(
        "\nResults saved to:"
    )

    print(
        run_folder
    )
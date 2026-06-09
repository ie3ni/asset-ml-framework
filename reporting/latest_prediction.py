# ============================================================
# LATEST PREDICTION MODULE
# ============================================================

def get_latest_prediction(
        model,
        scaler,
        X,
        df,
        persistence_accuracy):

    # ============================================================
    # LATEST FEATURES
    # ============================================================

    latest_features = X.iloc[-1:]

    latest_scaled = scaler.transform(latest_features)

    # ============================================================
    # MODEL PREDICTION
    # ============================================================

    latest_prediction = model.predict(latest_scaled)[0]

    # ============================================================
    # MODEL PROBABILITY (if available)
    # ============================================================

    try:
        probability_up = model.predict_proba(latest_scaled)[0][1]
        probability_text = f"Probability UP: {probability_up:.1%}"
    except AttributeError:
        probability_text = "Probability UP: N/A"

    # ============================================================
    # MODEL TEXT
    # ============================================================

    model_name = type(model).__name__

    if latest_prediction == 1:
        model_text = f"{model_name} Prediction: Asset likely UP tomorrow"
    else:
        model_text = f"{model_name} Prediction: Asset likely DOWN tomorrow"

    # ============================================================
    # ALWAYS UP BASELINE
    # ============================================================

    always_up_text = (
        "Always Up Baseline: Asset likely UP tomorrow\n"
        "Probability UP: 100.0%"
    )

    # ============================================================
    # PERSISTENCE BASELINE
    # ============================================================

    latest_direction = df["today_direction"].iloc[-1]

    if latest_direction == 1:
        persistence_text = (
            "Persistence Baseline: Asset likely UP tomorrow\n"
            f"Historical Accuracy: {persistence_accuracy:.1%}"
        )
    else:
        persistence_text = (
            "Persistence Baseline: Asset likely DOWN tomorrow\n"
            f"Historical Accuracy: {persistence_accuracy:.1%}"
        )

    # ============================================================
    # FINAL OUTPUT STRING
    # ============================================================

    prediction_text = (
        f"{model_text}\n"
        f"{probability_text}\n\n"
        f"{always_up_text}\n"
        f"{persistence_text}"
    )

    return prediction_text


# ============================================================
# PRINT FUNCTION (UNCHANGED)
# ============================================================

def print_latest_prediction(prediction_text):

    print("\n==============================")
    print(prediction_text)
    print("==============================")
# 📊Asset Machine Learning Research Framework

## Overview
DISCLAIMER: This project is intended for research purposes only and is in no way financial advice.

This project is a modular machine learning research pipeline for predicting an assets short term price direction using technical indicators and multiple models.

It compares:
- Machine Learning Models (Logistic Regression, Random Forest, etc.)
- Naive Baselines (Always Up, Persistence)
- Probabilistic outputs (model confidence)

The goal is not just prediction, but research-grade evaluation of predictive signal vs naive benchmarks. The architecture is modular and can be upgraded, as well as repurposed for other research uses. This version of the model requires fine-tuning through basic parameters and my current model has already evolved beyond this frameowkr but it still serves as a good guide and starting point.

*To generate and work on each model individually the user only needs to access the model_config.py file and uncomment the model they want to run, and then they can fine tune the desired parameters

*To switch the asset being analyzed same idea but in the asset_config.py file

---

# 🧠 Models Used

## Machine Learning Models (Active Model)

Examples:
- Logistic Regression
- Random Forest
- XGBOOST

These models output:
- A binary prediction (UP = 1, DOWN = 0)
- A probability estimate (if supported)

---

## 🔢 Model Output Interpretation

### ✔ Prediction Example

Model Prediction: Asset likely DOWN tomorrow

This means the model believes the probability of a downward move is higher than upward movement.

---

### 📈 Probability UP

Probability UP: 42.8%

This represents the model’s estimated probability that price will increase tomorrow.

---

## 📊 How to Interpret Probability

| Probability UP | Interpretation |
|----------------|----------------|
| ~50% | No clear signal (random-like behavior) |
| 55–60% | Weak predictive edge |
| 60–70% | Moderate signal strength |
| 70%+ | Strong (rare in financial markets) |

---

### ⚠️ Key Quant Insight

Even models with 45–55% accuracy may still have no tradable edge after transaction costs.

This is why baselines are essential.

---

# 📉 Baseline Models

## 1. Always Up Baseline

Always Up Baseline: Asset likely UP tomorrow  
Probability UP: 100.0%

### Interpretation

This assumes the asset always goes up tomorrow.

It is unrealistic but useful as a reference benchmark.

If your model cannot outperform this consistently, it has no directional edge.

---

## 2. Persistence Baseline

Persistence Baseline: Asset likely UP tomorrow  
Historical Accuracy: 53.7%

### What it means

This assumes tomorrow behaves like today:

- If price went up today → predict up tomorrow  
- If price went down today → predict down tomorrow  

### Interpretation

| Value | Meaning |
|------|--------|
| ~50% | No memory effect in price action |
| >50% | Short-term trend persistence |
| <50% | Mean reversion behavior |

---

# 📈 Model Comparison

Model Comparison:
Persistence     53.7%
Always Up       46.3%
Logistic Model  45.3%

### Interpretation

A model is only useful if:

Model Accuracy > Best Baseline

Otherwise, it adds no predictive value.

---

# 📊 Feature Importance

## Tree Models (Random Forest)
- Impurity-based importance

## Linear Models (Logistic Regression)
- Absolute coefficient magnitude

### Example Output

ma_spread → 0.15  
return_1d → 0.14  
rsi → 0.09  

### Interpretation

- Trend-following features (moving averages) dominate
- Momentum indicators contribute moderately
- Oscillators like RSI often weaker in noisy markets

---

# ⚠️ Limitations

- Financial markets are non-stationary
- Feature importance changes over time
- 50% accuracy can still be normal
- High accuracy does not guarantee profitability


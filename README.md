# 🩺 Diabetes Progression Predictor

A Streamlit web app that predicts **diabetes disease progression** one year after baseline, using a Ridge regression model trained on the scikit-learn Diabetes dataset.

🔗 **Live demo:** [diabetes-predictor-jjvtukn2rzkchbfkcdwrek.streamlit.app](https://diabetes-predictor-jjvtukn2rzkchbfkcdwrek.streamlit.app)

<!-- Add a screenshot: save it as screenshot.png in the repo, then uncomment -->
<!-- ![App screenshot](screenshot.png) -->

---

## What it predicts

The app outputs a single **progression score**: a measure of how much a patient's diabetes had progressed one year after their baseline measurements.

- In the dataset, scores range from about **25 to 346**, with an average of about **152**.
- A **higher** score means more progression.
- It is a regression output, not a probability, a diagnosis or a blood sugar value.

## Features

- 🎛️ **Slider-based input** in real units (years, kg/m², mmHg, mg/dL). No typing needed.
- ⚡ **Quick start patients** (Low / Medium / High / Random) taken from the real dataset.
- 🔮 **Predict button**: the result appears only after you press it.
- 📊 **Result panel** with a colour-coded score, a gauge, the percentile versus the dataset, and a chart of which inputs pushed the score up or down.
- 📂 **Batch mode**: drag and drop a CSV to predict many patients and download the results.
- ℹ️ **Sidebar** with model details, test performance, a feature guide and usage tips.

## Model

| Item | Value |
|---|---|
| Algorithm | Ridge regression (`alpha=1`, `solver="lsqr"`) |
| Pipeline | `StandardScaler` → `Ridge` |
| Data | scikit-learn Diabetes dataset (442 patients, 10 features) |
| Test split | 80/20, `random_state=42` |
| Test R² | ≈ 0.45 |
| Test RMSE | ≈ 53.8 |
| Test MAE | ≈ 42.8 |

The model was chosen after comparing multiple regression models with cross-validation and tuning the best one with GridSearchCV (see the notebook).

### Input features

| Code | Meaning |
|---|---|
| age | Age (years) |
| sex | Sex (coded 1 / 2 in the dataset) |
| bmi | Body mass index |
| bp | Average blood pressure |
| s1 | Total serum cholesterol (TC) |
| s2 | Low-density lipoproteins (LDL) |
| s3 | High-density lipoproteins (HDL) |
| s4 | Total cholesterol / HDL ratio (TCH) |
| s5 | Log of serum triglycerides (LTG) |
| s6 | Blood sugar level (GLU) |

The app shows friendly units and converts them to the scaled values the model was trained on.

## Project structure

```
├── app.py                         # Streamlit app
├── best_diabetes_model.joblib     # Trained model (StandardScaler + Ridge)
├── diabetes_regression_project.ipynb  # Training notebook
├── requirements.txt               # Python dependencies
└── .streamlit/
    └── config.toml                # Theme settings
```


## Tech stack

Python · Streamlit · scikit-learn · pandas · NumPy · Matplotlib · joblib

## ⚠️ Disclaimer

This is an **educational project**. It is **not** a medical device and must not be used for diagnosis or treatment decisions. Predictions are rough estimates based on a small, public dataset.

## Data source

Efron, Hastie, Johnstone and Tibshirani (2004), *Least Angle Regression*, Annals of Statistics, via `sklearn.datasets.load_diabetes`.


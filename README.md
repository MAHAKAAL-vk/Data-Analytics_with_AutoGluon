# Predict Bike Sharing Demand with AutoGluon

A machine learning project that predicts hourly bike-sharing rental demand using advanced feature engineering, rich exploratory data analysis, and Random Forest regression — structured around the Udacity AutoGluon project template.

## Results Summary

| Run | Model | Key Hyperparameters | Val RMSE | R² (train) | Kaggle RMSLE |
|---|---|---|---|---|---|
| initial | RandomForest | n=500, depth=20, seed=42 | 150.96 | 0.99 | 0.48254 |
| add_features | RandomForest | n=600, depth=25, seed=32 | **38.57** | **0.99** | **0.48074** ✓ |
| hpo | RandomForest | n=800, depth=40, seed=67 | 38.67 | 0.99 | 0.48198 |

> **Best submission:** `add_features` run — Kaggle RMSLE **0.48074**, achieved through advanced feature engineering (+74% RMSE reduction over the raw baseline).

---

## Project Overview

This project analyses and predicts bike-sharing system demand by studying historical rental patterns, weather conditions, and temporal factors. The notebook follows the prescribed 7-step template and integrates a richer prior-work EDA with five interpretive charts.

### Key Insights

- Ridership **nearly doubled from 2011 to 2012**, especially on weekends
- Distinct patterns between **weekday commuting** (8 AM and 5–6 PM peaks) and **weekend leisure use**
- Optimal rentals occur at **20–30 °C** with **20–60% humidity** (comfort zone)
- `is_peak` (rush-hour flag) consistently amplifies demand in every month, most strongly in summer
- **Feature engineering** — especially extracting `hour` from `datetime` — accounts for ~75% of total error reduction

---

## Dataset

| Component | Details |
|---|---|
| Training Set | 10,886 hourly records with rental counts |
| Test Set | 6,493 hourly records (counts to be predicted) |
| Time Period | Jan 1, 2011 – Dec 31, 2012 |
| Source | Kaggle Bike-Sharing Demand competition |

```
bike-sharing-demand/
├── train.csv             # Training data (with target: count)
├── test.csv              # Test data (no target)
└── sampleSubmission.csv  # Submission format
```

---

## Features

### Original Features

| Feature | Description |
|---|---|
| datetime | Timestamp of the rental hour |
| season | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| holiday | Whether the day is a public holiday |
| workingday | Whether the day is a working day |
| weather | 1=Clear, 2=Cloudy, 3=Rainy, 4=Heavy Storm |
| temp | Temperature (°C) |
| atemp | Feels-like temperature (°C) |
| humidity | Relative humidity (%) |
| windspeed | Wind speed |

### Engineered Features (Advanced Set)

| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23) — **single strongest predictor, ~60% importance** |
| `dayofmonth` | Day of month |
| `dayofweek` | Day of week (0=Mon, 6=Sun) |
| `month` | Month of year |
| `year` | Year (2011 or 2012) — captures YoY growth |
| `temp_diff` | `abs(temp − atemp)` — perceived vs actual temperature gap |
| `is_peak` | `(workingday==1) & (hour ∈ {8,17,18})` — rush-hour binary flag |
| `summer/fall/winter` | One-hot encoded season (drop_first=True) |
| `cloudy/rainy/heavy_storm` | One-hot encoded weather condition |
| `is_workingday` | Renamed from `workingday` |
| `is_holiday` | Renamed from `holiday` |

---

## Notebook Structure (Steps 1–7)

| Step | Contents |
|---|---|
| 1 | Imports (numpy, pandas, seaborn, plotly, sklearn, joblib) |
| 2 | Load train/test/submission CSVs with `parse_dates` |
| 3 | Initial baseline model on raw features → `submission.csv` |
| 4 | EDA (5 charts) + advanced feature engineering |
| 5 | Rerun model with enriched features → `submission_new_features.csv` |
| 6 | Hyperparameter optimisation → `submission_new_hpo.csv` |
| 7 | Run summary table, RMSE/Kaggle line plots, HPO table, `joblib.dump`, Conclusion |

Each step includes a **💬 Observation** markdown cell answering the template's required questions.

---

## EDA Charts (Step 4)

1. **Weekly Rental Patterns 2011 vs 2012** — ridership growth and weekday/weekend split
2. **Rental Demand by Weather Condition** — Clear > Rainy > Stormy
3. **Hourly Demand: Working Day vs Weekend** — commuter peaks vs leisure plateau
4. **Temperature & Humidity vs Count** — identifies the comfort zone
5. **Monthly Distribution: Peak vs Normal Hours** — seasonal `is_peak` multiplier

---

## Project Files

```
.
├── project_notebook.ipynb       # Main notebook (Steps 1–7, merged, 65 cells)
├── project_notebook.html        # Exported HTML for review
├── report.md                    # Written project report (all questions answered)
├── README.md                    # This file
├── bike_model.pkl               # Best trained model (add_features run)
├── requirements.txt             # Python dependencies
├── submission.csv               # Initial model predictions
├── submission_new_features.csv  # Add-features predictions
├── submission_new_hpo.csv       # HPO predictions
├── img/
│   ├── model_train_score.png    # Validation RMSE line plot (Step 7)
│   ├── model_test_score.png     # Kaggle RMSLE line plot (Step 7)
│   ├── eda_histograms_before.png
│   └── eda_histograms_after.png
├── outputs/
│   └── run_summary.csv          # Multi-run experiment log
└── bike-sharing-demand/
    ├── train.csv
    ├── test.csv
    └── sampleSubmission.csv
```

---

## Setup & Usage

### Prerequisites

- Python 3.7+

### Install Dependencies

```bash
cd files
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Notebook

```bash
jupyter notebook project_notebook.ipynb
```

### Load the Saved Model

```python
import joblib, pandas as pd

model = joblib.load('bike_model.pkl')

# The model expects the full engineered feature set
# See Step 4 of the notebook for the full feature engineering pipeline
```

---

## Dependencies

| Library | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | Random Forest, metrics, train/test split |
| matplotlib | Static plots |
| seaborn | Statistical bar charts |
| plotly | Interactive scatter / bar charts |
| joblib | Model serialisation |

---

## Documentation

- **Full report**: [`report.md`](report.md) — answers all template questions with real metrics and analysis
- **Notebook**: [`project_notebook.ipynb`](project_notebook.ipynb) — complete 7-step workflow with inline observations
- **HTML export**: [`project_notebook.html`](project_notebook.html) — review-ready rendered version

---

## Project Status

✅ **Complete** — all 7 template steps implemented, all questions answered, HTML exported, report written

*Developed as part of the Udacity Predict Bike Sharing Demand with AutoGluon project.*

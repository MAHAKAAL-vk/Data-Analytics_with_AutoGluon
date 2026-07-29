# Bike-Sharing Demand Prediction

A machine learning project that predicts hourly bike-sharing rental demand using advanced feature engineering and Random Forest regression.

## Project Overview

This project analyzes and predicts bike-sharing system demand by studying historical rental patterns, weather conditions, and temporal factors. The model achieved an **R² score of 0.99** and a **Mean Absolute Error (MAE) of 9.37**, demonstrating strong predictive performance.

### Key Insights

- Bike-sharing adoption nearly **doubled between 2011 and 2012**
- Distinct patterns emerge between **weekday commuting** (8 AM and 5 PM peaks) and **weekend leisure use**
- Optimal rental conditions occur at **20°C to 30°C** with **20-60% humidity**
- Peak hours consistently drive higher demand regardless of season

## Dataset

The project uses the **Bike-Sharing Demand Dataset** containing:

| Component        | Details                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| **Training Set** | 10,886 hourly records with rental counts                                         |
| **Test Set**     | 6,493 hourly records (counts to be predicted)                                    |
| **Time Period**  | Jan 1, 2011 - Dec 31, 2012                                                       |
| **Features**     | 11 original features (datetime, temperature, humidity, windspeed, weather, etc.) |

### Dataset Location

```
bike-sharing-demand/
├── train.csv          # Training data with target variable (count)
├── test.csv           # Test data without target variable
└── sampleSubmission.csv # Sample submission format
```

## Features & Feature Engineering

### Original Features

- **datetime**: Date and time of rental
- **season**: 1=Spring, 2=Summer, 3=Fall, 4=Winter
- **holiday**: Whether the day is a holiday
- **workingday**: Whether the day is a working day
- **weather**: 1=Clear, 2=Cloudy, 3=Rainy, 4=Heavy Storm
- **temp**: Temperature in Celsius
- **atemp**: "Feels like" temperature in Celsius
- **humidity**: Relative humidity percentage
- **windspeed**: Wind speed

### Engineered Features

- **hour**: Hour of day extracted from datetime
- **day**: Day of week (0-6)
- **month**: Month of year (1-12)
- **year**: Year (2011 or 2012)
- **temp_diff**: Absolute difference between actual and "feels like" temperature
- **is_peak**: Binary indicator for peak commuting hours (8 AM, 5-6 PM on working days)
- **is_workingday**: Renamed from workingday
- **is_holiday**: Renamed from holiday
- **weather_dummies**: One-hot encoded weather conditions (cloudy, rainy, heavy_storm)
- **season_dummies**: One-hot encoded seasons (summer, fall, winter)

## Model & Results

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Hyperparameters**:
  - n_estimators: 400 trees
  - max_depth: 50
  - n_jobs: -2 (parallel processing)
  - random_state: 52

### Performance Metrics (Training Set)

- **R² Score**: 0.99 (explains 99% of variance)
- **RMSE**: 15.03 (Root Mean Squared Error)
- **MAE**: 9.37 (Mean Absolute Error)

### Key Model Findings

The model effectively captures:

1. **Temporal Patterns**: Distinguishes between commuting (weekdays) and leisure (weekends) usage
2. **Seasonal Variations**: Accounts for demand changes across seasons
3. **Weather Effects**: Identifies optimal comfort zones for rentals
4. **Year-over-Year Growth**: Reflects adoption increase from 2011 to 2012

## Project Structure

```
.
├── README.md                          # This file
├── project_notebook.ipynb             # Main analysis and modeling notebook
├── requirements.txt                   # Python dependencies
├── bike_model.pkl                     # Trained Random Forest model
├── submission.csv                     # Final predictions
└── bike-sharing-demand/
    ├── train.csv                      # Training dataset
    ├── test.csv                       # Test dataset
    └── sampleSubmission.csv           # Sample submission format
```

## Installation & Setup

### Prerequisites

- Python 3.7+
- pip or conda package manager

### Step 1: Clone/Extract Project

```bash
cd /media/mahakaal/kali1/Udacit_Projects/Data-Analytics_with_AutoGluon
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Notebook

```bash
jupyter notebook project_notebook.ipynb
```

### Making Predictions

The trained model is saved as `bike_model.pkl`. To load and use it:

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('bike_model.pkl')

# Load test data
test = pd.read_csv('./bike-sharing-demand/test.csv')

# Make predictions
predictions = model.predict(test)
```

### Generating Submissions

The notebook generates predictions and saves them to `submission.csv` with the format:

```
datetime,count
2011-01-20 00:00:00,1
2011-01-20 01:00:00,2
...
```

## Dependencies

Core libraries used in this project:

| Library          | Purpose                          |
| ---------------- | -------------------------------- |
| **pandas**       | Data manipulation and analysis   |
| **numpy**        | Numerical computing              |
| **scikit-learn** | Machine learning (Random Forest) |
| **matplotlib**   | Static visualization             |
| **seaborn**      | Statistical visualization        |
| **plotly**       | Interactive visualization        |
| **joblib**       | Model serialization              |
| **autogluon**    | AutoML framework                 |

See `requirements.txt` for specific versions.

## Analysis Methodology

### 1. Exploratory Data Analysis (EDA)

- Dataset shape, info, and descriptive statistics
- Missing value analysis
- Temporal and seasonal pattern exploration
- Weather impact analysis
- Correlation between features and rental demand

### 2. Feature Engineering

- Time-based features (hour, day, month, year)
- Interaction features (peak hour detection, temperature difference)
- Categorical encoding (one-hot encoding for season and weather)
- Feature renaming for clarity

### 3. Model Training

- Train-test split using provided datasets
- Random Forest with optimized hyperparameters
- Training on full training set
- Validation using R², RMSE, and MAE metrics

### 4. Prediction & Submission

- Predictions on test set
- Constraint application (minimum count = 1)
- Submission file generation

## Key Conclusions

The analysis demonstrates that bike-sharing demand is multifactorial:

1. **Temporal Factors**: Peak hours during commuting times significantly increase demand
2. **Day Type**: Working days and weekends show distinctly different patterns
3. **Seasonal Trends**: Demand fluctuates across seasons with optimal conditions in summer
4. **Weather Impact**: Temperature and humidity create a "comfort zone" for users
5. **Growth Trend**: Year-over-year adoption increase affects overall demand levels

The Random Forest model successfully synthesizes these factors, achieving 99% accuracy on training data with minimal prediction error.

## Project Status

✅ **Complete** - Model trained, validated, and predictions generated

## Author

Developed as part of the Udacity Data Analytics with AutoGluon project

## License

This project uses the Bike-Sharing Demand dataset from UCI Machine Learning Repository.

---

**For questions or improvements, refer to the Jupyter notebook for detailed step-by-step analysis and visualizations.**

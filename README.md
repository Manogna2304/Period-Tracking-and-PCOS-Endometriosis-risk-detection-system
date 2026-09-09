# Period Tracking & PCOS/Endometriosis Risk Detection System

An ML-powered health analytics application designed to track menstrual cycles, estimate cycle length, and assess potential PCOS and endometriosis risk using symptom-based and structured health data.

## Features

* **Menstrual Cycle Tracking** – Track cycle-related information and estimate cycle length.
* **PCOS Risk Assessment** – Uses machine learning to identify patterns associated with PCOS risk.
* **Endometriosis Risk Assessment** – Evaluates symptom-based features to estimate potential endometriosis risk.
* **Symptom-Based Scoring** – Uses factors such as pain indicators, BMI, bleeding patterns, and cycle irregularities.
* **Machine Learning Models** – Classification models with feature engineering for predictive analysis.
* **Synthetic Data Fallback** – Provides a fallback data pipeline when required datasets are unavailable.
* **SQL Database** – Stores user information and health-tracking data for personalized analysis.
* **Interactive Dashboard** – Streamlit-based interface for entering data, viewing predictions, and visualizing insights.

## Tech Stack

* **Programming:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Database:** SQL
* **Frontend:** Streamlit
* **Visualization:** Matplotlib / Plotly

## Machine Learning Approach

The system uses structured health and symptom-related features for prediction, including:

* Age
* BMI
* Cycle length and irregularity
* Pelvic pain
* Heavy bleeding
* Pain during intercourse
* Family history
* Exercise patterns
* Pain scores

Feature engineering is applied before training classification models. A synthetic-data fallback pipeline is also included to maintain application functionality when the required training dataset is unavailable.

## Project Workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Machine Learning Model
    ↓
Risk Prediction
    ↓
Health Insights & Visualization
    ↓
SQL Database Storage
```

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <project-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

The application will open in your browser through the Streamlit interface.

## Project Structure

```text
├── app.py
├── models/
├── data/
├── database/
├── requirements.txt
└── README.md
```

> The exact structure may vary depending on the final organization of the project files.

## Important Note

This project is intended as an **educational and early-awareness tool**, not as a medical diagnostic system. The predictions should not be considered a substitute for professional medical evaluation or diagnosis.

## Future Improvements

* Train models on larger, clinically validated datasets.
* Improve model performance through hyperparameter tuning and cross-validation.
* Add model evaluation metrics and explainable AI features.
* Expand personalized cycle and symptom analytics.
* Deploy the application as a scalable web service.

## Author

**Manogna**

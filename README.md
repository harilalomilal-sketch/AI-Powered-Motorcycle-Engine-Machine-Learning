# AI-Powered Motorcycle Engine Health Prediction Using Machine Learning

A software-only predictive-maintenance prototype that analyzes engine sensor readings to classify engine condition (`Normal` or `Faulty`), estimate fault probability, highlight subsystem anomalies, and visualize sensor importance through an interactive Streamlit dashboard.

## Academic Scope & Limitation
This project uses a general **Engine Predictive Maintenance Dataset** rather than motorcycle-specific telemetry. It is designed and presented as a **motorcycle-oriented predictive-maintenance prototype**, and its outputs should not be treated as validated mechanical diagnoses.

## Dataset & Input Features
The model evaluates six core engine sensor parameters to predict `Engine Condition` (`0` = Normal, `1` = Faulty):
- **Engine rpm**
- **Lub oil pressure**
- **Fuel pressure**
- **Coolant pressure**
- **lub oil temp**
- **Coolant temp**

## Project Structure
```text
motorcycle-engine-health-prediction/
├── data/
│   ├── engine_data.csv          # Engine sensor dataset (7 columns)
│   └── DATASET_INFO.txt         # Instructions for swapping in the full 19,535-row dataset
├── models/
│   ├── best_engine_model.pkl    # Saved best-performing ML classifier
│   └── model_metrics.json       # Evaluation metrics across all 5 algorithms
├── src/
│   └── train.py                 # Preprocessing, 5-model training, evaluation & model export
├── app.py                       # Interactive Streamlit diagnostic dashboard
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## System Architecture
1. **Data Preprocessing (`src/train.py`):** Loads and cleans sensor records using `pandas` and splits data into stratified training and test sets (`80/20`).
2. **Multi-Algorithm Comparison:** Trains and benchmarks five classifiers (`Random Forest`, `Gradient Boosting`, `Decision Tree`, `K-Nearest Neighbors`, and `Logistic Regression`) across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
3. **Interactive Diagnostic Dashboard (`app.py`):** Built with `streamlit`, featuring single-reading prediction, subsystem threshold checks, Explainable AI feature importance, batch CSV log screening with downloadable reports, and a model comparison view.

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Train or retrain the model on `data/engine_data.csv`:
   ```bash
   python src/train.py
   ```
3. Launch the Streamlit web dashboard:
   ```bash
   streamlit run app.py
   ```

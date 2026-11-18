import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def main():
    data_path = os.path.join("data", "engine_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # 1. Load and clean dataset
    df = pd.read_csv(data_path).dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Define 5 ML algorithms for comparison
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    best_model = None
    best_f1 = -1.0
    best_name = ""
    comparison_results = []

    print("--- Model Comparison Results ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds).tolist()

        print(f"{name:22s} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        comparison_results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Confusion Matrix": cm,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nBest Selected Model: {best_name} (F1-Score: {best_f1:.4f})")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # 3. Save best model and evaluation metrics
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_engine_model.pkl")
    metrics_path = os.path.join("models", "model_metrics.json")

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "features": list(X.columns),
        },
        model_path,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "dataset_rows": int(len(df)),
                "features": list(X.columns),
                "results": comparison_results,
            },
            f,
            indent=2,
        )

    print(f"Saved trained model to {model_path}")
    print(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    main()

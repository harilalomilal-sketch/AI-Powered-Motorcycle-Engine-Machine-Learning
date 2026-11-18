import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Motorcycle Engine Health AI",
    layout="wide",
)

st.title("AI-Powered Motorcycle Engine Health Prediction")
st.caption(
    "Predictive-maintenance software prototype trained on engine sensor telemetry. "
    "Intended for academic demonstration and early anomaly screening."
)

MODEL_PATH = os.path.join("models", "best_engine_model.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run `python src/train.py` first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
model_name = artifact.get("model_name", "Selected Classifier")
feature_names = artifact["features"]


def analyze_subsystems(rpm, oil_p, fuel_p, cool_p, oil_t, cool_t):
    alerts = []
    if oil_p < 2.2 or oil_t > 88.0:
        alerts.append("Lubrication System: Low oil pressure or elevated oil temperature detected.")
    if fuel_p < 5.0 or fuel_p > 18.5:
        alerts.append("Fuel Delivery System: Fuel pressure outside normal operating band.")
    if cool_p < 1.0 or cool_t > 90.0:
        alerts.append("Cooling System: Abnormal coolant pressure or high coolant temperature.")
    if rpm < 600 or rpm > 11000:
        alerts.append("Engine Speed (RPM): Operating outside standard idle/cruise RPM range.")
    return alerts


tab1, tab2, tab3 = st.tabs([
    "Single Reading Diagnosis",
    "Batch CSV Log Analysis",
    "Model Evaluation & Metrics",
])

with tab1:
    st.subheader("Enter Engine Sensor Readings")
    col1, col2, col3 = st.columns(3)

    with col1:
        rpm = st.number_input("Engine RPM", min_value=0.0, max_value=25000.0, value=3000.0, step=50.0)
        oil_pressure = st.number_input("Lubricating Oil Pressure (bar)", min_value=0.0, max_value=20.0, value=3.5, step=0.1)

    with col2:
        fuel_pressure = st.number_input("Fuel Pressure (bar)", min_value=0.0, max_value=40.0, value=10.0, step=0.5)
        coolant_pressure = st.number_input("Coolant Pressure (bar)", min_value=0.0, max_value=15.0, value=2.0, step=0.1)

    with col3:
        oil_temp = st.number_input("Lubricating Oil Temperature (°C)", min_value=0.0, max_value=200.0, value=82.0, step=1.0)
        coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=200.0, value=85.0, step=1.0)

    if st.button("Predict Engine Condition", type="primary"):
        input_df = pd.DataFrame(
            [[rpm, oil_pressure, fuel_pressure, coolant_pressure, oil_temp, coolant_temp]],
            columns=feature_names,
        )
        prediction = int(model.predict(input_df)[0])
        if hasattr(model, "predict_proba"):
            fault_prob = float(model.predict_proba(input_df)[0][1]) * 100.0
        else:
            fault_prob = 100.0 if prediction == 1 else 0.0

        st.divider()
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            if prediction == 0:
                st.success(f"Engine Health: Normal (Active Model: {model_name})")
                st.info("Recommended Action: Continue routine monitoring of engine parameters.")
            else:
                st.error(f"Engine Health: Faulty / Maintenance Needed (Active Model: {model_name})")
                st.warning("Recommended Action: Inspect lubrication, fuel delivery, and cooling subsystems.")

            subsystem_alerts = analyze_subsystems(
                rpm, oil_pressure, fuel_pressure, coolant_pressure, oil_temp, coolant_temp
            )
            if subsystem_alerts:
                st.markdown("**Subsystem Diagnostic Notes:**")
                for alert in subsystem_alerts:
                    st.write(f"- {alert}")

        with res_col2:
            st.metric("Estimated Fault Probability", f"{fault_prob:.1f}%")

        if hasattr(model, "feature_importances_"):
            st.subheader("Explainable AI: Sensor Influence")
            importance_df = pd.DataFrame({
                "Sensor": feature_names,
                "Importance": model.feature_importances_,
            }).sort_values(by="Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Sensor"))

with tab2:
    st.subheader("Batch Engine Sensor Log Screening")
    st.write(
        "Upload a CSV file containing sensor logs to screen multiple records at once and download a diagnostic report."
    )
    uploaded_file = st.file_uploader("Upload Engine Sensor CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in feature_names if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
        else:
            X_batch = batch_df[feature_names].dropna()
            preds = model.predict(X_batch)
            probs = (
                model.predict_proba(X_batch)[:, 1] * 100.0
                if hasattr(model, "predict_proba")
                else preds * 100.0
            )

            report_df = batch_df.loc[X_batch.index].copy()
            report_df["Predicted_Condition"] = ["Faulty" if p == 1 else "Normal" for p in preds]
            report_df["Fault_Probability_%"] = [round(float(pr), 2) for pr in probs]

            total_rows = len(report_df)
            faulty_count = int((preds == 1).sum())
            normal_count = total_rows - faulty_count

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records Analyzed", total_rows)
            m2.metric("Normal Readings", normal_count)
            m3.metric("Faulty Readings Flagged", faulty_count)

            st.dataframe(report_df.head(50), use_container_width=True)

            csv_bytes = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Diagnostic Report (CSV)",
                data=csv_bytes,
                file_name="engine_health_diagnostic_report.csv",
                mime="text/csv",
            )

with tab3:
    st.subheader("5-Algorithm Benchmark Comparison")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        st.write(
            f"**Selected Production Model:** `{metrics_data['best_model']}` "
            f"(Trained on `{metrics_data['dataset_rows']}` records)"
        )
        metrics_df = pd.DataFrame(metrics_data["results"]).drop(columns=["Confusion Matrix"])
        st.dataframe(metrics_df, use_container_width=True)
        st.bar_chart(metrics_df.set_index("Model")[["Accuracy", "F1-Score", "ROC-AUC"]])
    else:
        st.info("Run `python src/train.py` to generate algorithm comparison metrics.")

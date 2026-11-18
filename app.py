import os
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="EngineAI Telemetry Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Monolith-Inspired Dark Engineering Cockpit CSS ---
st.markdown(
    """
    <style>
    /* Global dark navy backdrop */
    .stApp {
        background-color: #111927;
        color: #e2e8f0;
    }
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    /* Hide default Streamlit header/footer clutter */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Top Engineering Navigation Bar */
    .top-navbar {
        background: linear-gradient(90deg, #0d1420 0%, #162133 100%);
        border: 1px solid #26354d;
        border-radius: 6px;
        padding: 10px 18px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .brand-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .brand-badge {
        background-color: #ff4b4b;
        color: #ffffff;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 3px;
        letter-spacing: 0.08em;
    }
    .nav-meta {
        font-size: 0.78rem;
        color: #94a3b8;
    }

    /* Engineering Panel Cards */
    .eng-card {
        background-color: #182336;
        border: 1px solid #2a3b55;
        border-radius: 6px;
        padding: 12px 14px 10px 14px;
        margin-bottom: 10px;
    }
    .eng-card-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .eng-card-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        font-style: italic;
        margin-bottom: 8px;
    }
    .eng-section-label {
        font-size: 0.74rem;
        font-weight: 600;
        color: #cbd5e1;
        border-bottom: 1px solid #26354d;
        padding-bottom: 4px;
        margin-top: 6px;
        margin-bottom: 8px;
    }
    .param-label {
        font-size: 0.76rem;
        color: #cbd5e1;
        padding-top: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .value-pill {
        background-color: #101826;
        border: 1px solid #2d3f5e;
        border-radius: 4px;
        color: #f8fafc;
        font-family: monospace;
        font-size: 0.78rem;
        padding: 5px 8px;
        text-align: center;
        margin-top: 2px;
    }
    .filter-pill {
        background-color: #24324a;
        border: 1px solid #384c6e;
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 0.78rem;
        color: #f1f5f9;
        margin-top: 4px;
        margin-bottom: 6px;
    }
    .filter-caption {
        font-size: 0.70rem;
        color: #94a3b8;
    }
    .status-normal {
        background-color: rgba(16, 185, 129, 0.14);
        border: 1px solid #10b981;
        color: #6ee7b7;
        border-radius: 5px;
        padding: 8px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
        margin-top: 6px;
    }
    .status-faulty {
        background-color: rgba(255, 75, 75, 0.16);
        border: 1px solid #ff4b4b;
        color: #fca5a5;
        border-radius: 5px;
        padding: 8px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        text-align: center;
        margin-top: 6px;
    }

    /* Compact Streamlit widgets */
    div[data-testid="stSlider"] {
        padding-top: 2px;
        padding-bottom: 2px;
    }
    div[data-baseweb="select"] > div {
        background-color: #223047 !important;
        border-color: #35496b !important;
        font-size: 0.80rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = os.path.join("models", "best_engine_model.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")
DATA_PATH = os.path.join("data", "engine_data.csv")

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please run `python src/train.py` first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
model_name = artifact.get("model_name", "Random Forest")
feature_names = artifact["features"]


@st.cache_data
def load_dataset(path):
    if os.path.exists(path):
        return pd.read_csv(path).dropna().reset_index(drop=True)
    return None


df_samples = load_dataset(DATA_PATH)


def build_3d_engine_mesh(is_faulty: bool, rpm: float, coolant_temp: float) -> go.Figure:
    """Builds an interactive 3D motorcycle single-cylinder engine assembly visualization."""
    fig = go.Figure()

    # Crankcase (lower rounded housing)
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(-1.1, 1.1, 20)
    U, V = np.meshgrid(u, v)
    Xc = 1.25 * np.cos(U)
    Yc = V
    Zc = 1.05 * np.sin(U) - 0.8

    fig.add_trace(
        go.Surface(
            x=Xc,
            y=Yc,
            z=Zc,
            colorscale=[[0, "#64748b"], [1, "#cbd5e1"]],
            showscale=False,
            opacity=0.92,
            name="Crankcase",
            hoverinfo="name",
        )
    )

    # Cylinder barrel + cooling fins
    z_cyl = np.linspace(0.0, 2.2, 28)
    U_cyl, Z_cyl = np.meshgrid(u, z_cyl)
    fin_wave = 0.14 * (np.sin(Z_cyl * 16.0) > 0.2).astype(float)
    R_cyl = 0.78 + fin_wave
    X_cyl = R_cyl * np.cos(U_cyl)
    Y_cyl = R_cyl * np.sin(U_cyl)

    cyl_colors = (
        [[0, "#991b1b"], [1, "#ff4b4b"]]
        if (is_faulty or coolant_temp > 88.0)
        else [[0, "#7c8ba1"], [1, "#e2e8f0"]]
    )
    fig.add_trace(
        go.Surface(
            x=X_cyl,
            y=Y_cyl,
            z=Z_cyl,
            colorscale=cyl_colors,
            showscale=False,
            opacity=0.95,
            name="Cylinder & Cooling Fins",
            hoverinfo="name",
        )
    )

    # Cylinder head cover
    z_head = np.linspace(2.2, 2.75, 10)
    U_h, Z_h = np.meshgrid(u, z_head)
    X_h = 0.92 * np.cos(U_h)
    Y_h = 0.86 * np.sin(U_h)
    fig.add_trace(
        go.Surface(
            x=X_h,
            y=Y_h,
            z=Z_h,
            colorscale=[[0, "#475569"], [1, "#94a3b8"]],
            showscale=False,
            opacity=0.96,
            name="Cylinder Head",
            hoverinfo="name",
        )
    )

    # Exhaust header pipe curving forward and down
    t = np.linspace(0, 1, 30)
    ex_x = 0.75 + 1.15 * np.sin(t * np.pi * 0.65)
    ex_y = np.zeros_like(t)
    ex_z = 2.05 - 2.35 * (t ** 1.3)
    pipe_color = "#ff4b4b" if rpm > 9500 else "#cbd5e1"
    fig.add_trace(
        go.Scatter3d(
            x=ex_x,
            y=ex_y,
            z=ex_z,
            mode="lines",
            line=dict(color=pipe_color, width=12),
            name="Exhaust Header",
            hoverinfo="name",
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=265,
        paper_bgcolor="#182336",
        plot_bgcolor="#182336",
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#182336",
            camera=dict(eye=dict(x=1.65, y=1.45, z=1.1)),
            aspectmode="data",
        ),
    )
    return fig


def build_curve_prediction(
    selected_rpm: float,
    oil_p: float,
    fuel_p: float,
    cool_p: float,
    oil_t: float,
    cool_t: float,
) -> go.Figure:
    """Generates RPM vs Fault Probability curve with uncertainty band (Monolith style)."""
    rpm_grid = np.linspace(400, 12000, 60)
    grid_df = pd.DataFrame(
        {
            feature_names[0]: rpm_grid,
            feature_names[1]: np.full_like(rpm_grid, oil_p),
            feature_names[2]: np.full_like(rpm_grid, fuel_p),
            feature_names[3]: np.full_like(rpm_grid, cool_p),
            feature_names[4]: np.full_like(rpm_grid, oil_t),
            feature_names[5]: np.full_like(rpm_grid, cool_t),
        }
    )

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(grid_df)[:, 1] * 100.0
    else:
        probs = model.predict(grid_df) * 100.0

    # Estimate tree-ensemble uncertainty spread if Random Forest / Gradient Boosting
    if hasattr(model, "estimators_") and isinstance(model.estimators_, list):
        tree_preds = np.array([est.predict_proba(grid_df)[:, 1] * 100.0 for est in model.estimators_])
        std_band = np.std(tree_preds, axis=0) * 0.65
    else:
        std_band = np.full_like(probs, 6.5)

    upper = np.clip(probs + std_band, 0, 100)
    lower = np.clip(probs - std_band, 0, 100)

    # Current selected point
    sel_df = pd.DataFrame(
        [[selected_rpm, oil_p, fuel_p, cool_p, oil_t, cool_t]],
        columns=feature_names,
    )
    sel_prob = (
        float(model.predict_proba(sel_df)[0][1]) * 100.0
        if hasattr(model, "predict_proba")
        else float(model.predict(sel_df)[0]) * 100.0
    )

    fig = go.Figure()

    # Shaded uncertainty ribbon
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([rpm_grid, rpm_grid[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255, 75, 75, 0.24)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Predicted uncertainty",
        )
    )

    # Main predicted curve
    fig.add_trace(
        go.Scatter(
            x=rpm_grid,
            y=probs,
            mode="lines",
            line=dict(color="#ff4b4b", width=3),
            name="Predicted",
        )
    )

    # Selected operating point marker
    fig.add_trace(
        go.Scatter(
            x=[selected_rpm],
            y=[sel_prob],
            mode="markers",
            marker=dict(color="#38bdf8", size=10, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
            name="Selected operating point",
        )
    )

    fig.update_layout(
        height=230,
        margin=dict(l=42, r=15, t=12, b=34),
        paper_bgcolor="#182336",
        plot_bgcolor="#141e30",
        font=dict(color="#cbd5e1", size=11),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.96,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(17, 25, 39, 0.65)",
            font=dict(size=10),
        ),
        xaxis=dict(
            title="Engine RPM Sweep",
            gridcolor="#25344d",
            zerolinecolor="#25344d",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Fault Risk (%)",
            range=[0, 105],
            gridcolor="#25344d",
            zerolinecolor="#25344d",
            tickfont=dict(size=10),
        ),
    )
    return fig


def build_scalar_gauge(title: str, value: float, min_val: float, max_val: float, suffix: str = "") -> go.Figure:
    """Creates a compact semicircular red-arc gauge matching the right-hand scalar prediction panel."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 17, "color": "#f8fafc"}},
            title={"text": title, "font": {"size": 11, "color": "#cbd5e1"}},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [min_val, max_val],
                    "tickwidth": 1,
                    "tickcolor": "#64748b",
                    "tickfont": {"size": 8, "color": "#94a3b8"},
                },
                "bar": {"color": "#ff4b4b", "thickness": 0.72},
                "bgcolor": "#26354d",
                "borderwidth": 0,
                "threshold": {
                    "line": {"color": "#f8fafc", "width": 3},
                    "thickness": 0.85,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=142,
        margin=dict(l=18, r=18, t=28, b=6),
        paper_bgcolor="#182336",
        font={"color": "#e2e8f0"},
    )
    return fig


# --- Top Engineering Navigation Header ---
st.markdown(
    f"""
    <div class="top-navbar">
        <div class="brand-title">
            <span class="brand-badge">ML STUDIO</span>
            <span>MOTO-ENGINE TELEMETRY &amp; DIAGNOSTICS</span>
        </div>
        <div class="nav-meta">
            Active Model: <b>{model_name}</b> &nbsp;|&nbsp; Sensors: <b>6 Channels</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

view_mode = st.radio(
    "Workspace View",
    ["Simulation Cockpit", "Batch CSV Log Analysis", "Model Evaluation & Metrics"],
    horizontal=True,
    label_visibility="collapsed",
)

if view_mode == "Simulation Cockpit":
    # Initialize default sensor values in session state
    defaults = {
        "rpm": 1450,
        "oil_p": 3.50,
        "fuel_p": 10.50,
        "cool_p": 2.20,
        "oil_t": 81.0,
        "cool_t": 82.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Three-column Monolith engineering layout: Left (Filter + 3D), Center (Sliders + Curve), Right (3 Gauges)
    left_col, center_col, right_col = st.columns([1.05, 1.55, 0.85], gap="small")

    # ================= LEFT COLUMN =================
    with left_col:
        st.markdown(
            """
            <div class="eng-card">
                <div class="eng-card-header">Filter Category</div>
                <div class="eng-card-sub">Using data Engine Sensor Telemetry + Benchmark Logs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        profile_options = ["Custom Manual Telemetry"]
        if df_samples is not None and len(df_samples) > 0:
            profile_options += [
                f"engine-{i:03d} ({'Faulty' if int(df_samples.iloc[i, -1]) == 1 else 'Normal'})"
                for i in range(min(30, len(df_samples)))
            ]

        selected_profile = st.selectbox(
            "Select Telemetry Record",
            profile_options,
            label_visibility="collapsed",
        )

        if selected_profile != "Custom Manual Telemetry" and df_samples is not None:
            row_idx = int(selected_profile.split("-")[1].split(" ")[0])
            row = df_samples.iloc[row_idx]
            st.session_state["rpm"] = int(np.clip(row.iloc[0], 300, 12000))
            st.session_state["oil_p"] = float(np.clip(row.iloc[1], 0.0, 10.0))
            st.session_state["fuel_p"] = float(np.clip(row.iloc[2], 0.0, 25.0))
            st.session_state["cool_p"] = float(np.clip(row.iloc[3], 0.0, 10.0))
            st.session_state["oil_t"] = float(np.clip(row.iloc[4], 40.0, 130.0))
            st.session_state["cool_t"] = float(np.clip(row.iloc[5], 40.0, 130.0))

        total_records = len(df_samples) if df_samples is not None else 800
        st.markdown(
            f"""
            <div class="filter-pill">{selected_profile}</div>
            <div class="filter-caption">Filtered from {total_records} to 1 active telemetry state</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="eng-card" style="margin-top: 10px;">
                <div class="eng-card-header">3D Engine Viewer Lookup</div>
                <div class="eng-card-sub">Interactive CAD thermal &amp; mechanical state mesh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ================= CENTER COLUMN =================
    with center_col:
        st.markdown(
            f"""
            <div class="eng-card">
                <div class="eng-card-header">Fix Parameters</div>
                <div class="eng-card-sub">Using model {model_name}. Generating live sensitivity response</div>
                <div class="eng-section-label">Operating &amp; Fluid Pressure Parameters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def slider_row(label: str, key: str, min_v, max_v, step_v, fmt: str):
            c_lbl, c_sld, c_val = st.columns([1.35, 2.4, 0.75], gap="small")
            with c_lbl:
                st.markdown(f'<div class="param-label">{label}</div>', unsafe_allow_html=True)
            with c_sld:
                val = st.slider(
                    label,
                    min_value=min_v,
                    max_value=max_v,
                    value=st.session_state[key],
                    step=step_v,
                    key=f"sld_{key}",
                    label_visibility="collapsed",
                )
                st.session_state[key] = val
            with c_val:
                st.markdown(f'<div class="value-pill">{fmt.format(val)}</div>', unsafe_allow_html=True)
            return val

        rpm = slider_row("Engine RPM", "rpm", 300, 12000, 50, "{:d}")
        oil_p = slider_row("Lub Oil Pressure", "oil_p", 0.0, 10.0, 0.1, "{:.2f}")
        fuel_p = slider_row("Fuel Pressure", "fuel_p", 0.0, 25.0, 0.2, "{:.2f}")
        cool_p = slider_row("Coolant Pressure", "cool_p", 0.0, 10.0, 0.1, "{:.2f}")

        st.markdown('<div class="eng-section-label">Thermal Sensor Properties</div>', unsafe_allow_html=True)
        oil_t = slider_row("Lub Oil Temp (°C)", "oil_t", 40.0, 130.0, 0.5, "{:.1f}")
        cool_t = slider_row("Coolant Temp (°C)", "cool_t", 40.0, 130.0, 0.5, "{:.1f}")

    # Run live ML inference on current slider state
    input_df = pd.DataFrame(
        [[rpm, oil_p, fuel_p, cool_p, oil_t, cool_t]],
        columns=feature_names,
    )
    prediction = int(model.predict(input_df)[0])
    fault_prob = (
        float(model.predict_proba(input_df)[0][1]) * 100.0
        if hasattr(model, "predict_proba")
        else (100.0 if prediction == 1 else 0.0)
    )
    is_faulty = prediction == 1

    # Render 3D Viewer inside Left Column now that prediction state is computed
    with left_col:
        fig_3d = build_3d_engine_mesh(is_faulty=is_faulty, rpm=rpm, coolant_temp=cool_t)
        st.plotly_chart(fig_3d, use_container_width=True, config={"displayModeBar": False})

    # Render Curve Prediction inside Center Column
    with center_col:
        st.markdown(
            f"""
            <div class="eng-card" style="margin-top: 8px;">
                <div class="eng-card-header">Curve Prediction</div>
                <div class="eng-card-sub">Using model {model_name} across continuous Engine RPM sweep</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_curve = build_curve_prediction(rpm, oil_p, fuel_p, cool_p, oil_t, cool_t)
        st.plotly_chart(fig_curve, use_container_width=True, config={"displayModeBar": False})

    # ================= RIGHT COLUMN =================
    with right_col:
        st.markdown(
            f"""
            <div class="eng-card">
                <div class="eng-card-header">Scalar Prediction</div>
                <div class="eng-card-sub">Using model {model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Compute derived engineering indices for the 3 scalar dials
        lub_index = float(np.clip((oil_p / 5.0) * 6.0 + max(0.0, (95.0 - oil_t) / 15.0) * 4.0, 0.0, 10.0))
        thermal_load = float(np.clip((cool_t * 0.55 + oil_t * 0.45), 0.0, 130.0))

        st.plotly_chart(
            build_scalar_gauge("Fault Probability", round(fault_prob, 1), 0, 100, "%"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            build_scalar_gauge("Lubrication Index", round(lub_index, 2), 0, 10, ""),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            build_scalar_gauge("Thermal Load (°C)", round(thermal_load, 1), 40, 130, ""),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        if not is_faulty:
            st.markdown(
                '<div class="status-normal">ENGINE HEALTH: NORMAL<br><span style="font-weight:400;font-size:0.70rem;">Parameters within operating envelope</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-faulty">ENGINE HEALTH: FAULTY<br><span style="font-weight:400;font-size:0.70rem;">Inspect oil, fuel &amp; cooling systems</span></div>',
                unsafe_allow_html=True,
            )

elif view_mode == "Batch CSV Log Analysis":
    st.subheader("Batch Engine Sensor Log Screening")
    st.write("Upload a CSV file containing sensor logs to screen multiple records at once and download a diagnostic report.")
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

elif view_mode == "Model Evaluation & Metrics":
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

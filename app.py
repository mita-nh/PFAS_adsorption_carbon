"""Local Streamlit interface for the PFAS adsorption model."""

from __future__ import annotations

import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors


st.set_page_config(page_title="PFAS adsorption predictor", page_icon="🧪", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --ink: #16343b;
      --muted: #60787d;
      --teal: #0b746b;
      --teal-dark: #09554f;
      --mint: #e8f4f0;
      --paper: #f5f8f7;
      --line: #dce8e5;
      --gold: #edb75a;
    }
    [data-testid="stAppViewContainer"] { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #edf4f1; border-right: 1px solid var(--line); }
    .block-container { max-width: 1420px; padding-top: 1.5rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.025em; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    .hero {
      background: linear-gradient(118deg, #103d45 0%, #0b746b 72%, #159487 100%);
      color: #fff; border-radius: 22px; padding: 30px 34px 28px; margin: 0 0 26px;
      box-shadow: 0 12px 28px rgba(16, 61, 69, .14); position: relative; overflow: hidden;
    }
    .hero:after { content: ""; position: absolute; width: 220px; height: 220px; border: 1px solid rgba(255,255,255,.15); border-radius: 50%; right: 8%; top: -115px; box-shadow: 0 0 0 28px rgba(255,255,255,.035), 0 0 0 58px rgba(255,255,255,.025); }
    .hero-kicker { color: #aee1d5; font-size: .76rem; letter-spacing: .16em; font-weight: 700; text-transform: uppercase; margin-bottom: 9px; }
    .hero h1 { color: white; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.08; margin: 0 0 9px; }
    .hero p { color: #e0f0ed; font-size: 1.02rem; margin: 0; max-width: 780px; }
    .hero-chip { display: inline-block; margin-top: 17px; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); border-radius: 99px; color: white; padding: 6px 12px; font-size: .78rem; }
    .section-label { color: var(--teal); font-size: .76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin: 3px 0 5px; }
    .section-title { color: var(--ink); font-size: 1.55rem; font-weight: 750; letter-spacing: -.025em; margin: 0 0 3px; }
    .section-help { color: var(--muted); margin: 0 0 17px; font-size: .94rem; }
    div[data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 18px; padding: 22px 22px 16px; box-shadow: 0 6px 20px rgba(22,52,59,.045); }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px; box-shadow: 0 5px 16px rgba(22,52,59,.045); }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--teal-dark); }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button { background: var(--teal); color: #fff; border: 0; border-radius: 10px; font-weight: 700; min-height: 2.7rem; transition: .15s ease; }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { background: var(--teal-dark); color: #fff; border: 0; transform: translateY(-1px); }
    div[data-testid="stDownloadButton"] > button { border-radius: 10px; border-color: #b8d4ce; color: var(--teal-dark); font-weight: 650; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stFileUploaderDropzone"] { background: #fff; border-color: #bfd6d1; border-radius: 12px; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: var(--line); }
    @media (max-width: 700px) { .hero { padding: 24px 22px; } .block-container { padding-top: 1rem; } }
    </style>
    <div class="hero">
      <div class="hero-kicker">PFAS · ADSORPTION · MACHINE LEARNING</div>
      <h1>PFAS Adsorption Prediction Studio</h1>
      <p>Estimate equilibrium adsorption capacity (qe) of PFAS adsorption in carbon-based adsorbent, check whether conditions sit within the model’s applicability domain, and explore predicted adsorption isotherms.</p>
      <span class="hero-chip">Optimized model&nbsp;&nbsp; · &nbsp;&nbsp;Data Distance Index (DDI) Calculation&nbsp;&nbsp; · &nbsp;&nbsp;Langmuir + Freundlich Isotherm Simulation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Preserve the exact training-CSV order after applying the DROP_COLS list.
# The temperature header is retained as read by the training script (cp1252).
FEATURES = [
    "WPSA-3",
    "AMW",
    "SHsOH",
    "SsOH",
    "SddssS",
    "SsF",
    "PFAS concentration (mg/L)",
    "surface BET (m2/g)",
    "Particle diameter (mm)",
    "Batch dose (mg/L)",
    "Salt (mol/L)",
    "Temperature (??C)",
    "pH",
    "Stirring speed (rpm)",
    "Duration (hours)",
    "Single (0) or multicomponent (1)",
]

DESCRIPTOR_FEATURES = ["WPSA-3", "AMW", "SHsOH", "SsOH", "SddssS", "SsF"]
DDI_THRESHOLD = 1.8
K_NEIGHBORS = 5

DISPLAY = {
    "PFAS concentration (mg/L)": "Initial PFAS concentration (mg/L)",
    "surface BET (m2/g)": "BET surface area (m²/g)",
    "Particle diameter (mm)": "Particle diameter (mm)",
    "Batch dose (mg/L)": "Adsorbent dosage (mg/L)",
    "Salt (mol/L)": "Salt concentration (mol/L)",
    "Temperature (??C)": "Temperature (°C)",
    "pH": "pH",
    "Stirring speed (rpm)": "Stirring speed (rpm)",
    "Duration (hours)": "Contact duration (hours)",
}


@st.cache_resource(show_spinner=False)
def load_pickle(data: bytes):
    return joblib.load(io.BytesIO(data))


def freundlich(ce, kf, inv_n):
    return kf * np.power(ce, inv_n)


def langmuir(ce, qmax, kl):
    return qmax * kl * ce / (1.0 + kl * ce)


def predict_qe(model, y_scaler, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    normalized_ln_qe = np.asarray(model.predict(frame[FEATURES]), dtype=float).reshape(-1, 1)
    ln_qe = y_scaler.inverse_transform(normalized_ln_qe).ravel()
    # The training target is ln Qe, so exponentiation returns Qe in mg/g.
    qe = np.exp(ln_qe)
    return ln_qe, qe


def fit_isotherms(ce: np.ndarray, qe: np.ndarray) -> dict:
    fits = {}
    if len(ce) < 3:
        return fits
    for name, function, initial, bounds in [
        ("Freundlich", freundlich, [max(float(qe.max()), 1e-6), 0.5], ([0, 0], [np.inf, np.inf])),
        ("Langmuir", langmuir, [max(float(qe.max()), 1e-6), 0.1], ([0, 0], [np.inf, np.inf])),
    ]:
        try:
            params, _ = curve_fit(function, ce, qe, p0=initial, bounds=bounds, maxfev=50000)
            predicted = function(ce, *params)
            fits[name] = {
                "function": function,
                "params": params,
                "r2": float(r2_score(qe, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(qe, predicted))),
            }
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            fits[name] = {"error": str(exc)}
    return fits


st.title("PFAS adsorption predictor")
st.caption("Machine learning prediction of qe with applicability-domain screening and fitted adsorption isotherms.")
st.info(
    "This prototype expects the model and DDI assets as uploads. DDI is computed from the same 14 model "
    "features, scaled with your saved DDI scaler, then compared with the supplied cutoff of 1.8."
)

with st.sidebar:
    st.header("Model assets")
    st.markdown("### ⚙️ Setup")
    st.caption("Load the model bundle and the DDI reference assets to enable predictions.")
    st.markdown("**Prediction assets**")
    model_file = st.file_uploader("Model file (.pkl)", type=["pkl", "joblib"], key="model")
    y_scaler_file = st.file_uploader("Target scaler (.pkl)", type=["pkl", "joblib"], key="yscaler")
    descriptor_file = st.file_uploader("PFAS descriptor table (.xlsx or .csv)", type=["xlsx", "csv"], key="descriptors")
    st.divider()
    st.subheader("Applicability-domain assets")
    st.markdown("**Applicability-domain assets**")
    ddi_scaler_file = st.file_uploader("DDI feature scaler (.pkl)", type=["pkl", "joblib"], key="ddiscaler")
    ddi_reference_file = st.file_uploader(
        "Raw DDI training features (.csv; 14 columns in model feature order)", type=["csv"], key="ddiref"
    )
    st.caption(f"DDI cutoff: {DDI_THRESHOLD:.2f} · k = {K_NEIGHBORS}")
    st.caption(f"Cutoff: DDI ≤ {DDI_THRESHOLD:.2f}  ·  k = {K_NEIGHBORS}")

missing = [
    label
    for label, file in [
        ("CatBoost model", model_file),
        ("target scaler", y_scaler_file),
        ("descriptor table", descriptor_file),
        ("DDI scaler", ddi_scaler_file),
        ("raw DDI reference CSV", ddi_reference_file),
    ]
    if file is None
]
if missing:
    st.warning("Upload the required assets in the sidebar to enable predictions: " + ", ".join(missing) + ".")
    st.markdown('<div class="section-label">Getting started</div><div class="section-title">Add your model assets</div><p class="section-help">Use the setup panel in the sidebar. The app will unlock once all five files are loaded.</p>', unsafe_allow_html=True)
    st.stop()

try:
    model = load_pickle(model_file.getvalue())
    y_scaler = load_pickle(y_scaler_file.getvalue())
    ddi_scaler = load_pickle(ddi_scaler_file.getvalue())
    if descriptor_file.name.lower().endswith(".xlsx"):
        descriptors = pd.read_excel(io.BytesIO(descriptor_file.getvalue()))
    else:
        descriptors = pd.read_csv(io.BytesIO(descriptor_file.getvalue()))
    descriptors.columns = [str(c).strip() for c in descriptors.columns]
    descriptors["Name"] = descriptors["Name"].astype(str).str.strip()
    reference = pd.read_csv(io.BytesIO(ddi_reference_file.getvalue()))
    if reference.shape[1] != len(FEATURES):
        raise ValueError(f"DDI reference CSV must have {len(FEATURES)} numeric columns; found {reference.shape[1]}.")
    reference_values = reference.to_numpy(dtype=float)
    if reference_values.shape[0] < K_NEIGHBORS:
        raise ValueError(f"DDI reference CSV must contain at least {K_NEIGHBORS} rows.")
    if not set(DESCRIPTOR_FEATURES + ["Name"]).issubset(descriptors.columns):
        raise ValueError("Descriptor table must contain Name and all four used descriptor columns: " + ", ".join(DESCRIPTOR_FEATURES))
    if descriptors["Name"].duplicated().any():
        raise ValueError("Descriptor table contains duplicate PFAS names.")
    if getattr(model, "feature_count_", len(FEATURES)) != len(FEATURES):
        raise ValueError(f"Model expects {model.feature_count_} features, but this app is configured for {len(FEATURES)}.")
    if getattr(y_scaler, "n_features_in_", 1) != 1:
        raise ValueError("The target scaler must be fitted to one target column (ln Qe).")
    # The upload must contain the unscaled training features; apply the same
    # fitted scaler to both the reference rows and new inputs.
    scaled_reference = ddi_scaler.transform(reference_values)
    neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(scaled_reference)
except Exception as exc:
    st.error(f"Could not load or validate the supplied assets: {exc}")
    st.stop()

st.subheader("Prediction inputs")
st.markdown('<div class="section-label">01 · Predict</div><div class="section-title">Define an experiment</div><p class="section-help">Choose a PFAS and enter the experimental conditions. The four molecular descriptors are filled from your descriptor table.</p>', unsafe_allow_html=True)
pfas_names = sorted(descriptors["Name"].dropna().unique().tolist())
with st.form("prediction_inputs"):
    pfas = st.selectbox("PFAS", pfas_names)
    col1, col2, col3 = st.columns(3)
    values = {}
    with col1:
        values["PFAS concentration (mg/L)"] = st.number_input(DISPLAY["PFAS concentration (mg/L)"], min_value=0.0, value=10.0, format="%.5f")
        values["surface BET (m2/g)"] = st.number_input(DISPLAY["surface BET (m2/g)"], min_value=0.0, value=100.0, step=1.0)
        values["Particle diameter (mm)"] = st.number_input(DISPLAY["Particle diameter (mm)"], min_value=0.0, value=0.1, format="%.4f")
    with col2:
        values["Batch dose (mg/L)"] = st.number_input(DISPLAY["Batch dose (mg/L)"], min_value=0.0, value=1000.0, format="%.5f")
        values["Salt (mol/L)"] = st.number_input(DISPLAY["Salt (mol/L)"], min_value=0.0, value=0.0, format="%.5f")
        values["Temperature (??C)"] = st.number_input(DISPLAY["Temperature (??C)"], value=25.0, step=1.0)
    with col3:
        values["pH"] = st.number_input(DISPLAY["pH"], min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        values["Stirring speed (rpm)"] = st.number_input(DISPLAY["Stirring speed (rpm)"], min_value=0.0, value=150.0, step=10.0)
        values["Duration (hours)"] = st.number_input(DISPLAY["Duration (hours)"], min_value=0.0, value=24.0, step=1.0)
    mixture = st.radio("System", ["Single-component", "Multicomponent"], horizontal=True)
    submit = st.form_submit_button("Predict qe")

if submit:
    descriptor = descriptors.loc[descriptors["Name"] == pfas].iloc[0]
    row = {name: float(descriptor[name]) for name in DESCRIPTOR_FEATURES}
    row.update(values)
    row["Single (0) or multicomponent (1)"] = 0 if mixture == "Single-component" else 1
    input_frame = pd.DataFrame([{name: row[name] for name in FEATURES}], columns=FEATURES)
    try:
        _, qe = predict_qe(model, y_scaler, input_frame)
        ddi_input = ddi_scaler.transform(input_frame[FEATURES].to_numpy(dtype=float))
        distances, _ = neighbors.kneighbors(ddi_input)
        ddi = float(distances.mean(axis=1)[0])
        within = ddi <= DDI_THRESHOLD
        left, right = st.columns([1, 1])
        left.metric("Predicted qe", f"{qe[0]:.4g} mg/g")
        right.metric("Applicability domain · DDI", f"{ddi:.4f}", "Within domain" if within else "Outside domain")
        if not within:
            st.warning("This input is outside the configured applicability domain (DDI > 1.8). Treat the prediction with caution.")
        st.session_state["last_inputs"] = row
        st.session_state["last_pfas"] = pfas
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Predicted adsorption isotherm")
st.write("Sweep initial concentration while holding the other entered conditions and PFAS constant.")
st.markdown('<div class="section-label" style="margin-top:2.5rem">02 · Explore</div><div class="section-title">Build a predicted isotherm</div><p class="section-help">Sweep initial concentration while keeping the selected PFAS and other experiment conditions fixed.</p>', unsafe_allow_html=True)
last = st.session_state.get("last_inputs")
last_pfas = st.session_state.get("last_pfas", pfas_names[0])
with st.form("isotherm_sweep"):
    c1, c2, c3 = st.columns(3)
    with c1:
        c0_min = st.number_input("Initial concentration minimum (mg/L)", min_value=0.0, value=0.1, step=0.0001, key="cmin", format="%.5f")
    with c2:
        c0_max = st.number_input("Initial concentration maximum (mg/L)", min_value=0.0001, value=100.0, step=0.0001, key="cmax", format="%.5f")
    with c3:
        n_points = st.number_input("Number of points", min_value=3, max_value=200, value=25, step=1)
    sweep_button = st.form_submit_button("Calculate and fit isotherm")

if sweep_button:
    if last is None:
        st.error("Submit a single prediction first so the app has the PFAS and fixed experimental conditions to use.")
    elif c0_max <= c0_min:
        st.error("Maximum concentration must be greater than minimum concentration.")
    else:
        c0_values = np.linspace(c0_min, c0_max, int(n_points))
        sweep = pd.DataFrame([{**last, "PFAS concentration (mg/L)": c0} for c0 in c0_values], columns=FEATURES)
        try:
            _, qe_values = predict_qe(model, y_scaler, sweep)
            # Batch dose is mg/L; divide by 1000 to express adsorbent dose in g/L.
            ce_values = c0_values - qe_values * sweep["Batch dose (mg/L)"].to_numpy() / 1000.0
            ddi_values = neighbors.kneighbors(ddi_scaler.transform(sweep[FEATURES].to_numpy(dtype=float)))[0].mean(axis=1)
            result = pd.DataFrame({
                "Initial concentration C0 (mg/L)": c0_values,
                "Equilibrium concentration Ce (mg/L)": ce_values,
                "Predicted qe (mg/g)": qe_values,
                "DDI": ddi_values,
                "Within domain": ddi_values <= DDI_THRESHOLD,
            })
            valid = np.isfinite(ce_values) & np.isfinite(qe_values) & (ce_values > 0) & (qe_values >= 0) & (ce_values <= c0_values)
            fit_mask = valid & (ddi_values <= DDI_THRESHOLD)
            fits = fit_isotherms(ce_values[fit_mask], qe_values[fit_mask])
            fig = go.Figure()
            for is_inside, label, color, symbol in [
                (True, "Within domain", "#18794e", "circle"),
                (False, "Outside domain", "#c43c39", "x"),
            ]:
                mask = valid & ((ddi_values <= DDI_THRESHOLD) == is_inside)
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=ce_values[mask], y=qe_values[mask], mode="markers", name=label,
                        marker={"color": color, "symbol": symbol, "size": 9},
                        customdata=np.column_stack([c0_values[mask], ddi_values[mask]]),
                        hovertemplate="Ce=%{x:.4g} mg/L<br>qe=%{y:.4g} mg/g<br>C0=%{customdata[0]:.4g} mg/L<br>DDI=%{customdata[1]:.4f}<extra></extra>",
                    ))
            if valid.sum() != len(valid):
                st.warning(f"{len(valid) - int(valid.sum())} points have physically invalid Ce/Qe and are omitted from the plot and fitting.")
            if fit_mask.sum() < 3:
                st.warning("Fewer than three physically valid, in-domain points are available; isotherm fits require at least three.")
            else:
                smooth_ce = np.linspace(float(ce_values[fit_mask].min()), float(ce_values[fit_mask].max()), 250)
                for name, fit in fits.items():
                    if "error" in fit:
                        st.warning(f"{name} fit did not converge: {fit['error']}")
                        continue
                    fig.add_trace(go.Scatter(x=smooth_ce, y=fit["function"](smooth_ce, *fit["params"]), mode="lines", name=f"{name} fit"))
                a, b = st.columns(2)
                f = fits.get("Freundlich", {})
                if "params" in f:
                    kf, inv_n = f["params"]
                    a.metric("Freundlich fit", f"Kf = {kf:.4g}; 1/n = {inv_n:.4g}; n = {1.0 / inv_n:.4g}")
                    a.caption(f"R² = {f['r2']:.3f} · RMSE = {f['rmse']:.4g} mg/g")
                l = fits.get("Langmuir", {})
                if "params" in l:
                    qmax, kl = l["params"]
                    b.metric("Langmuir fit", f"qmax = {qmax:.4g} mg/g; KL = {kl:.4g} L/mg")
                    b.caption(f"R² = {l['r2']:.3f} · RMSE = {l['rmse']:.4g} mg/g")
            fig.update_layout(
                title=f"{last_pfas} predicted adsorption isotherm",
                xaxis_title="Equilibrium concentration Ce (mg/L)",
                yaxis_title="Predicted qe (mg/g)",
                template="plotly_white",
                legend_title="Predictions / fits",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("Download sweep results (CSV)", result.to_csv(index=False).encode("utf-8"), "pfas_isotherm_predictions.csv", "text/csv")
            st.caption("Fits use only physically valid points that are within the DDI threshold. Freundlich requires Ce > 0.")
        except Exception as exc:
            st.error(f"Isotherm calculation failed: {exc}")

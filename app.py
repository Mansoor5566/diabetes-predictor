import inspect
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺", layout="wide")

MODEL_PATH = Path(__file__).parent / "best_diabetes_model.joblib"

# Works on old and new Streamlit versions (new: width="stretch", old: use_container_width)
STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}

# column, label, unit, step, slider format, csv column name
SPEC = {
    "age": ("Age", "years", 1.0, "%.0f", "age"),
    "bmi": ("BMI", "kg/m²", 0.1, "%.1f", "bmi"),
    "bp": ("Blood pressure", "mmHg", 0.5, "%.1f", "blood_pressure"),
    "s1": ("Total cholesterol", "mg/dL", 1.0, "%.0f", "total_cholesterol"),
    "s2": ("LDL (bad) cholesterol", "mg/dL", 0.1, "%.1f", "ldl"),
    "s3": ("HDL (good) cholesterol", "mg/dL", 1.0, "%.0f", "hdl"),
    "s4": ("Cholesterol ratio", "total / HDL", 0.1, "%.1f", "chol_ratio"),
    "s5": ("Triglycerides", "mg/dL", 1.0, "%.0f", "triglycerides"),
    "s6": ("Blood sugar", "mg/dL", 1.0, "%.0f", "glucose"),
}
SHORT = {"age": "Age", "sex": "Sex", "bmi": "BMI", "bp": "Blood pressure", "s1": "Total chol.",
         "s2": "LDL", "s3": "HDL", "s4": "Chol. ratio", "s5": "Triglycerides", "s6": "Blood sugar"}
CSV_COLS = ["age", "sex"] + [v[4] for k, v in SPEC.items() if k != "age"]
GREEN, AMBER, RED = "#2ca58d", "#f0a202", "#d1495b"


# ----------------------------------------------------------------- data/model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference():
    """Raw (real-unit) and scaled versions of the dataset + the exact raw->scaled mapping."""
    raw = load_diabetes(as_frame=True, scaled=False).frame
    scaled = load_diabetes(as_frame=True).frame
    maps = {c: np.polyfit(raw[c], scaled[c], 1) for c in raw.columns if c != "target"}
    return raw, scaled, maps


model = load_model()
raw_df, scaled_df, MAPS = load_reference()
ORDER = list(getattr(model, "feature_names_in_", [c for c in scaled_df.columns if c != "target"]))
target = raw_df["target"]
T_MIN, T_MAX = float(target.min()), float(target.max())
Q33, Q66 = float(target.quantile(1 / 3)), float(target.quantile(2 / 3))


def to_display(raw):
    """Dataset units -> what the sliders show (triglycerides in mg/dL instead of log)."""
    out = raw.copy()
    out["s5"] = np.exp(out["s5"])
    return out


def to_model_input(display):
    """Slider/CSV units -> the scaled features the model was trained on."""
    d = pd.DataFrame(display).copy()
    d["s5"] = np.log(d["s5"])
    for c, (a, b) in MAPS.items():
        d[c] = a * d[c] + b
    return d[ORDER]


def contributions(X):
    """Per-feature push on the prediction (works for scaler + linear model pipelines)."""
    try:
        z = model.steps[0][1].transform(X)[0]
        return pd.Series(model.steps[-1][1].coef_ * z, index=X.columns)
    except Exception:
        return None


def category(score):
    if score < Q33:
        return "Lower progression", GREEN
    if score < Q66:
        return "Moderate progression", AMBER
    return "Higher progression", RED


@st.cache_data
def test_metrics():
    """Reproduces the notebook's 80/20 split (random_state=42) and scores the saved model on the test part."""
    X = scaled_df[ORDER]
    y = scaled_df["target"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = model.predict(X_te)
    return {
        "r2": float(r2_score(y_te, pred)),
        "rmse": float(mean_squared_error(y_te, pred) ** 0.5),
        "mae": float(mean_absolute_error(y_te, pred)),
        "n": int(len(X_te)),
    }


def model_details():
    steps = " → ".join(type(step).__name__ for _, step in model.steps)
    est = model.steps[-1][1]
    params = {k: getattr(est, k) for k in ("alpha", "solver", "fit_intercept") if hasattr(est, k)}
    return steps, type(est).__name__, params


def coefficients():
    est = model.steps[-1][1]
    if not hasattr(est, "coef_"):
        return None
    return pd.Series(est.coef_, index=ORDER).rename(SHORT)


# -------------------------------------------------------------- slider state
def bounds(col):
    step = SPEC[col][2]
    series = to_display(raw_df)[col]
    return float(np.floor(series.min() / step) * step), float(np.ceil(series.max() / step) * step)


def snap(col, v):
    step = SPEC[col][2]
    lo, hi = bounds(col)
    return float(np.clip(round(round(v / step) * step, 4), lo, hi))


def apply_row(row):
    """Put a patient (display units) into the widgets."""
    for col in SPEC:
        st.session_state[f"ui_{col}"] = snap(col, float(row[col]))
    st.session_state["ui_sex"] = int(row["sex"])
    st.session_state.pop("result", None)


def preset(q):
    disp = to_display(raw_df)
    idx = (target - target.quantile(q)).abs().idxmin()
    apply_row(disp.loc[idx])


def random_patient():
    apply_row(to_display(raw_df).iloc[np.random.randint(len(raw_df))])


if "ui_age" not in st.session_state:
    apply_row(to_display(raw_df).median())
    st.session_state["ui_sex"] = 1


def current_inputs():
    vals = {col: st.session_state[f"ui_{col}"] for col in SPEC}
    vals["sex"] = st.session_state["ui_sex"]
    return vals


def reset_inputs():
    apply_row(to_display(raw_df).median())
    st.session_state["ui_sex"] = 1


# -------------------------------------------------------------------- visuals
def gauge(score):
    cmap = LinearSegmentedColormap.from_list("risk", [GREEN, AMBER, RED])
    fig, ax = plt.subplots(figsize=(6, 0.9))
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), extent=[T_MIN, T_MAX, 0, 1], aspect="auto", cmap=cmap)
    x = float(np.clip(score, T_MIN, T_MAX))
    ax.axvline(x, color="#1f2937", lw=3)
    ax.scatter([x], [1.2], marker="v", s=150, color="#1f2937", clip_on=False)
    ax.set_xlim(T_MIN, T_MAX)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_alpha(0)
    return fig


def contrib_chart(contrib):
    c = contrib.rename(SHORT).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(c.index, c.values, color=[RED if v > 0 else GREEN for v in c.values])
    ax.axvline(0, color="#9ca3af", lw=1)
    ax.set_xlabel("pushes score  ← down   |   up →")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    fig.tight_layout()
    return fig


st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .stButton > button {border-radius: 12px; font-weight: 600;}
      div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Diabetes Progression Predictor")

# ==================================================================== sidebar
with st.sidebar:
    st.header("ℹ️ Model info")

    steps, est_name, params = model_details()
    st.markdown(f"**Model:** {est_name} regression  \n**Pipeline:** {steps}")
    st.caption("Predicts a number (regression), not a yes/no class.")

    try:
        m = test_metrics()
        st.markdown("**Test performance**")
        m1, m2 = st.columns(2)
        m1.metric("R²", f"{m['r2']:.2f}")
        m2.metric("RMSE", f"{m['rmse']:.1f}")
        st.metric("MAE", f"{m['mae']:.1f}")
        st.caption(
            f"Scored on {m['n']} held-out patients (80/20 split, random_state=42, same as the notebook). "
            "R² = share of variation explained. RMSE / MAE = typical error in score points."
        )
    except Exception:
        st.caption("Performance numbers unavailable.")

    with st.expander("⚙️ Settings & data"):
        st.markdown(
            "\n".join(
                [f"- **{k}:** {v}" for k, v in params.items()]
                + [
                    "- **Scaling:** StandardScaler (inside the pipeline)",
                    f"- **Features:** {len(ORDER)}",
                    f"- **Training data:** scikit-learn Diabetes dataset, {len(raw_df)} patients",
                    f"- **Score range:** {T_MIN:.0f} – {T_MAX:.0f} (average {target.mean():.0f})",
                    f"- **scikit-learn (running):** {sklearn.__version__}",
                ]
            )
        )

    coefs = coefficients()
    if coefs is not None:
        with st.expander("📈 What the model relies on"):
            st.caption("Coefficients (per 1 standard deviation). Positive pushes the score up, negative down.")
            st.bar_chart(coefs.sort_values())

    with st.expander("🧪 Feature guide"):
        st.markdown(
            "- **Age**: years\n"
            "- **Sex**: coded 1 / 2 in the dataset\n"
            "- **BMI**: body mass index\n"
            "- **Blood pressure**: average (mean arterial)\n"
            "- **Total cholesterol** (s1): all cholesterol in blood\n"
            "- **LDL** (s2): \"bad\" cholesterol\n"
            "- **HDL** (s3): \"good\" cholesterol\n"
            "- **Cholesterol ratio** (s4): total ÷ HDL\n"
            "- **Triglycerides** (s5): blood fats (dataset stores the log)\n"
            "- **Blood sugar** (s6): glucose level"
        )

    with st.expander("❓ How to use"):
        st.markdown(
            "1. Pick a **quick start** patient or move the sliders.\n"
            "2. Press **Predict**.\n"
            "3. Read the score, the gauge and what moved it.\n\n"
            "Many patients at once? Use the **Drop a CSV** tab."
        )

    st.button("🔄 Reset inputs", on_click=reset_inputs, **STRETCH)

    st.divider()
    st.caption(
        "**What the score means:** disease progression one year after baseline. "
        "Higher = more progression. The Lower / Moderate / Higher badge only compares with this dataset."
    )
    st.warning("Educational project — not medical advice or a diagnosis.", icon="⚠️")
    st.caption("Data: Efron et al., *Least Angle Regression* (2004), via scikit-learn.")

tab_single, tab_batch = st.tabs(["🎛️  Predict", "📂  Drop a CSV"])

# ============================================================ single patient
with tab_single:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.caption("⚡ Quick start")
        b1, b2, b3, b4 = st.columns(4)
        b1.button("🟢 Low", on_click=preset, args=(0.10,), **STRETCH)
        b2.button("🟡 Medium", on_click=preset, args=(0.50,), **STRETCH)
        b3.button("🔴 High", on_click=preset, args=(0.90,), **STRETCH)
        b4.button("🎲 Random", on_click=random_patient, **STRETCH)

        def slider(col, container):
            label, unit, step, fmt, _ = SPEC[col]
            lo, hi = bounds(col)
            container.slider(f"{label} ({unit})", lo, hi, step=step, format=fmt, key=f"ui_{col}")

        with st.container(border=True):
            st.markdown("##### 👤 Profile")
            c1, c2 = st.columns(2)
            slider("age", c1)
            c2.radio("Sex", [1, 2], horizontal=True, key="ui_sex",
                     format_func=lambda v: f"Type {v}", help="Coded 1 / 2 in the original dataset.")
            slider("bmi", c1)
            slider("bp", c2)

        with st.container(border=True):
            st.markdown("##### 🩸 Blood tests")
            c1, c2 = st.columns(2)
            for i, col in enumerate(["s1", "s2", "s3", "s4", "s5", "s6"]):
                slider(col, c1 if i % 2 == 0 else c2)

        if st.button("🔮  Predict", type="primary", **STRETCH):
            inputs = current_inputs()
            X = to_model_input(pd.DataFrame([inputs]))
            score = float(model.predict(X)[0])
            st.session_state["result"] = {
                "inputs": inputs,
                "score": score,
                "contrib": contributions(X),
            }

    with right:
        res = st.session_state.get("result")
        with st.container(border=True):
            if res is None:
                st.markdown(
                    "<div style='text-align:center;padding:5rem 0;color:#9ca3af'>"
                    "<div style='font-size:3.5rem'>🎯</div>"
                    "<div style='font-size:1.1rem'>Set the sliders, then press <b>Predict</b></div></div>",
                    unsafe_allow_html=True,
                )
            else:
                if res["inputs"] != current_inputs():
                    st.warning("Inputs changed — press Predict to refresh.", icon="⚠️")
                score = res["score"]
                label, color = category(score)
                pct = float((target < score).mean() * 100)
                st.markdown(
                    f"""
                    <div style='text-align:center'>
                      <div style='color:#6b7280'>Predicted progression score</div>
                      <div style='font-size:4.5rem;font-weight:800;line-height:1.1;color:{color}'>{score:.0f}</div>
                      <span style='background:{color};color:white;padding:4px 14px;border-radius:999px;font-weight:600'>{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.pyplot(gauge(score), **STRETCH)
                st.caption(f"Higher than **{pct:.0f}%** of patients in the dataset  ·  range {T_MIN:.0f} – {T_MAX:.0f}")
                if res["contrib"] is not None:
                    st.markdown("**What moved the score**")
                    st.pyplot(contrib_chart(res["contrib"]), **STRETCH)

    st.caption("Educational project — not medical advice.")

# ===================================================================== batch
with tab_batch:
    st.markdown("Drop a CSV with one patient per row and get a prediction for each.")
    template = to_display(raw_df).drop(columns="target").iloc[[0, 1, 2]].round(2)
    template.columns = CSV_COLS
    template["sex"] = template["sex"].astype(int)
    st.download_button("⬇️ Download template", template.to_csv(index=False), "template.csv", "text/csv")

    up = st.file_uploader("Drag & drop CSV here", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up)
            missing = [c for c in CSV_COLS if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                inv = {v[4]: k for k, v in SPEC.items()}
                disp = data[CSV_COLS].rename(columns=inv)
                scores = model.predict(to_model_input(disp))
                out = data.copy()
                out["predicted_score"] = np.round(scores, 1)
                out["level"] = [category(s)[0] for s in scores]
                st.success(f"Predicted {len(out)} patients")
                st.dataframe(out, **STRETCH, hide_index=True)
                st.download_button("⬇️ Download results", out.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read that file: {e}")
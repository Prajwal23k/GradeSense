# ============================================================
# app.py — Streamlit app using the Stacking Ensemble only
# Run: streamlit run app.py
# ============================================================

import os, pickle, warnings
import numpy as np
import pandas as pd
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import config
from feature_engineering import add_engineered_features

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.2rem; margin: 0 0 8px 0; }
.hero p  { opacity: 0.75; margin: 0; font-size: 1rem; }

.result-pass {
    background: linear-gradient(135deg, #0f9b58, #27ae60);
    border-radius: 14px; padding: 28px 32px;
    color: white; text-align: center;
    box-shadow: 0 8px 24px rgba(39,174,96,0.35);
}
.result-fail {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 14px; padding: 28px 32px;
    color: white; text-align: center;
    box-shadow: 0 8px 24px rgba(231,76,60,0.35);
}
.result-pass h2, .result-fail h2 { font-size: 2rem; margin: 0 0 6px 0; }
.result-pass p,  .result-fail p  { opacity: 0.88; margin: 0; font-size: 1rem; }

.prob-card {
    background: #f8f9fc;
    border-radius: 12px; padding: 18px 22px;
    text-align: center; border: 1px solid #e2e6ea;
}
.prob-card .label { font-size: 0.78rem; color: #666; text-transform: uppercase;
                    letter-spacing: .08em; margin-bottom: 4px; }
.prob-card .value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }

.badge {
    display: inline-block;
    background: linear-gradient(90deg,#667eea,#764ba2);
    color: white; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: .06em; text-transform: uppercase;
    margin-bottom: 12px;
}

.section-title {
    font-size: 1.1rem; font-weight: 600;
    color: #1a1a2e; margin: 24px 0 12px 0;
    border-left: 4px solid #667eea;
    padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Load ensemble ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_ensemble():
    import tensorflow as tf

    with open(os.path.join(config.MODEL_DIR, "ensemble.pkl"), "rb") as f:
        parts = pickle.load(f)

    ann = tf.keras.models.load_model(
        os.path.join(config.MODEL_DIR, "ensemble_ann.keras")
    )

    with open(os.path.join(config.MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(config.MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    ensemble = {
        "base_learners": parts["base_learners"],
        "ann":           ann,
        "meta_learner":  parts["meta_learner"],
    }
    return ensemble, scaler, feature_names


def build_input_vector(raw_input, scaler, feature_names):
    df = pd.DataFrame([raw_input])
    df = add_engineered_features(df)
    for col in config.LEAKAGE_COLS + ["pass"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return scaler.transform(df.values)


def run_ensemble(ensemble, X):
    cols = [m.predict_proba(X)[:, 1] for m in ensemble["base_learners"].values()]
    cols.append(ensemble["ann"].predict(X, verbose=0).flatten())
    X_meta = np.column_stack(cols)
    prob   = ensemble["meta_learner"].predict_proba(X_meta)[:, 1][0]
    return float(prob)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar_inputs():
    st.sidebar.markdown("## 📋 Student Profile")

    st.sidebar.markdown("**📚 Academics**")
    studytime = st.sidebar.selectbox(
        "Weekly Study Time",
        [1,2,3,4],
        format_func=lambda x:{1:"< 2 hrs",2:"2–5 hrs",3:"5–10 hrs",4:"> 10 hrs"}[x],
    )
    failures = st.sidebar.slider("Past Class Failures", 0, 3, 0)
    absences = st.sidebar.slider("Number of Absences", 0, 93, 5)

    st.sidebar.markdown("**👤 Personal**")
    age      = st.sidebar.slider("Age", 15, 22, 17)
    sex      = st.sidebar.radio("Sex", ["F","M"], horizontal=True)
    address  = st.sidebar.radio("Address", ["U","R"],
                                 format_func=lambda x:"Urban" if x=="U" else "Rural",
                                 horizontal=True)
    famsize  = st.sidebar.radio("Family Size", ["GT3","LE3"],
                                 format_func=lambda x:"> 3 members" if x=="GT3" else "≤ 3 members",
                                 horizontal=True)
    Pstatus  = st.sidebar.radio("Parents Status", ["T","A"],
                                 format_func=lambda x:"Together" if x=="T" else "Apart",
                                 horizontal=True)
    romantic = st.sidebar.radio("In Relationship", ["yes","no"], horizontal=True)

    st.sidebar.markdown("**🏫 School**")
    school    = st.sidebar.radio("School", ["GP","MS"], horizontal=True)
    schoolsup = st.sidebar.radio("School Support", ["yes","no"], horizontal=True)
    famsup    = st.sidebar.radio("Family Support", ["yes","no"], horizontal=True)
    paid      = st.sidebar.radio("Paid Extra Classes", ["yes","no"], horizontal=True)
    activities= st.sidebar.radio("Extracurricular", ["yes","no"], horizontal=True)
    internet  = st.sidebar.radio("Internet at Home", ["yes","no"], horizontal=True)
    higher    = st.sidebar.radio("Wants Higher Edu", ["yes","no"], horizontal=True)
    nursery   = st.sidebar.radio("Attended Nursery", ["yes","no"], horizontal=True)
    reason    = st.sidebar.selectbox("School Choice Reason",
                                      ["course","home","other","reputation"])
    traveltime= st.sidebar.selectbox("Travel Time",
                                      [1,2,3,4],
                                      format_func=lambda x:{1:"< 15 min",2:"15–30 min",
                                                            3:"30–60 min",4:"> 1 hr"}[x])

    st.sidebar.markdown("**👨‍👩‍👧 Parents**")
    Medu     = st.sidebar.slider("Mother Education (0–4)", 0, 4, 2)
    Fedu     = st.sidebar.slider("Father Education (0–4)", 0, 4, 2)
    Mjob     = st.sidebar.selectbox("Mother's Job",
                                     ["at_home","health","other","services","teacher"])
    Fjob     = st.sidebar.selectbox("Father's Job",
                                     ["at_home","health","other","services","teacher"])
    guardian = st.sidebar.selectbox("Guardian", ["mother","father","other"])

    st.sidebar.markdown("**🎭 Social & Lifestyle**")
    freetime = st.sidebar.slider("Free Time (1–5)", 1, 5, 3)
    goout    = st.sidebar.slider("Goes Out (1–5)", 1, 5, 3)
    Dalc     = st.sidebar.slider("Weekday Alcohol (1–5)", 1, 5, 1)
    Walc     = st.sidebar.slider("Weekend Alcohol (1–5)", 1, 5, 2)
    health   = st.sidebar.slider("Health Status (1–5)", 1, 5, 3)
    famrel   = st.sidebar.slider("Family Relations (1–5)", 1, 5, 4)

    return dict(
        school=school, sex=sex, age=age, address=address,
        famsize=famsize, Pstatus=Pstatus, Medu=Medu, Fedu=Fedu,
        Mjob=Mjob, Fjob=Fjob, reason=reason, guardian=guardian,
        traveltime=traveltime, studytime=studytime, failures=failures,
        schoolsup=schoolsup, famsup=famsup, paid=paid,
        activities=activities, nursery=nursery, higher=higher,
        internet=internet, romantic=romantic, famrel=famrel,
        freetime=freetime, goout=goout, Dalc=Dalc, Walc=Walc,
        health=health, absences=absences,
    )


# ── Risk breakdown ────────────────────────────────────────────────────────────
def risk_factors(raw_input):
    flags = []
    if raw_input["failures"] >= 2:
        flags.append(("🔴", "High", f"{raw_input['failures']} past failures"))
    elif raw_input["failures"] == 1:
        flags.append(("🟡", "Medium", "1 past failure"))

    if raw_input["absences"] > 15:
        flags.append(("🔴", "High", f"{raw_input['absences']} absences (high)"))
    elif raw_input["absences"] > 8:
        flags.append(("🟡", "Medium", f"{raw_input['absences']} absences"))

    if raw_input["studytime"] == 1:
        flags.append(("🔴", "High", "Very low study time (< 2 hrs/week)"))
    elif raw_input["studytime"] == 2:
        flags.append(("🟡", "Medium", "Below-average study time (2–5 hrs/week)"))

    if raw_input["Dalc"] + raw_input["Walc"] >= 7:
        flags.append(("🔴", "High", "High total alcohol consumption"))
    elif raw_input["Dalc"] + raw_input["Walc"] >= 5:
        flags.append(("🟡", "Medium", "Moderate alcohol consumption"))

    if raw_input["higher"] == "no":
        flags.append(("🟡", "Medium", "No interest in higher education"))

    if not flags:
        flags.append(("🟢", "Low", "No major risk factors detected"))

    return flags


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Hero header
    st.markdown("""
    <div class="hero">
        <div class="badge">AI-Powered · Stacking Ensemble</div>
        <h1>🎓 Student Performance Predictor</h1>
        <p>Combines Random Forest · XGBoost · Logistic Regression · Neural Network
           into one intelligent prediction engine.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        ensemble, scaler, feature_names = load_ensemble()
    except FileNotFoundError:
        st.error("⚠️  Models not found. Run `python train_all.py` first.")
        st.stop()

    raw_input = sidebar_inputs()

    # Predict on every change (live prediction)
    X = build_input_vector(raw_input, scaler, feature_names)
    prob_pass = run_ensemble(ensemble, X)
    prob_fail = 1.0 - prob_pass
    prediction = "PASS" if prob_pass >= 0.5 else "FAIL"
    confidence = max(prob_pass, prob_fail) * 100

    # ── Result card ────────────────────────────────────────────
    col_result, col_gap, col_detail = st.columns([2, 0.15, 1.85])

    with col_result:
        if prediction == "PASS":
            st.markdown(f"""
            <div class="result-pass">
                <h2>✅ PASS</h2>
                <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fail">
                <h2>❌ FAIL</h2>
                <p>Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Probability Breakdown</div>",
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <div class="prob-card">
            <div class="label">Pass Probability</div>
            <div class="value" style="color:#27ae60">{prob_pass*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="prob-card">
            <div class="label">Fail Probability</div>
            <div class="value" style="color:#e74c3c">{prob_fail*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        # Gauge bar
        st.markdown("<br>", unsafe_allow_html=True)
        bar_color = "#27ae60" if prediction == "PASS" else "#e74c3c"
        st.markdown(f"""
        <div style="background:#eee;border-radius:20px;height:14px;margin-top:4px">
          <div style="width:{prob_pass*100:.1f}%;background:{bar_color};
                      height:14px;border-radius:20px;transition:width .4s ease">
          </div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:.75rem;color:#888;margin-top:4px">
          <span>0% PASS</span><span>50%</span><span>100% PASS</span>
        </div>
        """, unsafe_allow_html=True)

    with col_detail:
        st.markdown("<div class='section-title'>Risk Factor Analysis</div>",
                    unsafe_allow_html=True)
        for icon, level, desc in risk_factors(raw_input):
            st.markdown(f"{icon} **{level}** — {desc}")

        st.markdown("<div class='section-title'>Key Inputs</div>",
                    unsafe_allow_html=True)
        summary = {
            "Study Time": {1:"<2h",2:"2–5h",3:"5–10h",4:">10h"}[raw_input["studytime"]],
            "Failures": raw_input["failures"],
            "Absences": raw_input["absences"],
            "Higher Edu": raw_input["higher"],
            "Internet": raw_input["internet"],
            "Family Relations": raw_input["famrel"],
            "Alcohol (W+D)": raw_input["Dalc"] + raw_input["Walc"],
        }
        for k, v in summary.items():
            st.markdown(f"- **{k}**: {v}")

    # ── Model performance section ─────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📊 Model Performance Comparison</div>",
                unsafe_allow_html=True)

    results_path = os.path.join(config.OUTPUT_DIR, "model_results.csv")
    if os.path.exists(results_path):
        df_res = pd.read_csv(results_path).sort_values("accuracy", ascending=False)
        df_res.index = range(1, len(df_res)+1)

        # Highlight ensemble row
        def highlight_ensemble(row):
            if "Ensemble" in str(row["model"]):
                return ["background-color: #e8f5e9; font-weight: bold"] * len(row)
            return [""] * len(row)

        df_display = df_res[["model","accuracy","roc_auc"]].copy()
        df_display["accuracy"] = df_display["accuracy"].map("{:.4f}".format)
        df_display["roc_auc"]  = df_display["roc_auc"].map("{:.4f}".format)
        st.dataframe(df_display.style.apply(highlight_ensemble, axis=1),
                     use_container_width=True)

    comp_img = os.path.join(config.OUTPUT_DIR, "model_comparison.png")
    roc_img  = os.path.join(config.OUTPUT_DIR, "roc_curves.png")
    if os.path.exists(comp_img) and os.path.exists(roc_img):
        c1, c2 = st.columns(2)
        c1.image(comp_img, caption="Accuracy & AUC Comparison", use_container_width=True)
        c2.image(roc_img,  caption="ROC Curves",               use_container_width=True)

    # ── EDA plots ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-title'>📈 Data Insights</div>",
                unsafe_allow_html=True)
    eda = {
        "Grade Distribution":      "01_grade_distribution.png",
        "Feature Correlations":    "02_correlation_heatmap.png",
        "Study Time vs Pass Rate": "03_studytime_vs_passrate.png",
        "Failures vs Pass Rate":   "04_failures_vs_passrate.png",
    }
    cols = st.columns(2)
    for i, (title, fname) in enumerate(eda.items()):
        path = os.path.join(config.OUTPUT_DIR, fname)
        if os.path.exists(path):
            cols[i%2].image(path, caption=title, use_container_width=True)

    # ANN training history
    ann_hist = os.path.join(config.OUTPUT_DIR, "ann_training_history.png")
    if os.path.exists(ann_hist):
        st.image(ann_hist, caption="ANN Training History", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<small>Stacking Ensemble: Random Forest · XGBoost · "
        "Logistic Regression · Neural Network</small>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

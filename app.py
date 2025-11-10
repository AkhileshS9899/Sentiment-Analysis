import os
import pickle
import joblib
import numpy as np
import streamlit as st

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="💬 Sentiment Predictor", page_icon="💬", layout="centered")

PRIMARY = "#4F46E5"
POS = "#16a34a"
NEU = "#64748b"
NEG = "#dc2626"

# ----------------------
# Custom CSS Styling
# ----------------------
st.markdown(
    f"""
    <style>
    .main {{
        padding-top: 1rem;
        text-align: center;
    }}
    .title-box {{
        background: linear-gradient(135deg, {PRIMARY}22, {PRIMARY}11);
        border: 1px solid {PRIMARY}33;
        padding: 18px 22px;
        border-radius: 16px;
        margin-bottom: 25px;
    }}
    .pill {{
        display:inline-block; padding:8px 14px; border-radius:999px; font-weight:600;
        border:1px solid #ffffff22; font-size:1rem; margin:6px 0;
    }}
    .prob-row {{
        display:flex; gap:10px; align-items:center; margin:6px 0;
    }}
    .bar {{
        height: 10px; border-radius: 999px; background: #ffffff18; flex:1; overflow:hidden;
    }}
    .fill {{ height:100%; border-radius:999px; }}
    .explainer {{
        font-size:0.95rem; opacity:0.9; line-height:1.5;
        background:#ffffff08; padding:14px 16px; border-radius:12px; border:1px solid #ffffff12;
        margin-top:20px;
    }}
    .history-item {{
        border:1px solid #ffffff22; border-radius:12px; padding:10px 14px; margin:6px 0; 
        display:flex; justify-content:space-between; align-items:center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Header
# ----------------------
st.markdown(
    f"""
    <div class="title-box">
        <h1 style="margin:0">💬 Sentiment Predictor</h1>
        <div style="opacity:.85">Analyze your text and instantly see if it’s <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b>.</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Load Model Automatically
# ----------------------
def load_pipeline():
    """Try loading the trained model automatically."""
    for fname in ["final_sentiment_model.pkl", "mode.pkl"]:
        if os.path.exists(fname):
            try:
                return joblib.load(fname), f"✅ Model Loaded: {fname}"
            except Exception:
                with open(fname, "rb") as f:
                    return pickle.load(f), f"✅ Model Loaded: {fname}"
    return None, "⚠️ Model file not found. Please ensure final_sentiment_model.pkl exists in this folder."

pipeline, load_msg = load_pipeline()
st.markdown(f"<p style='text-align:center; color:#9ca3af;'>{load_msg}</p>", unsafe_allow_html=True)

# Detect classifier name
model_name = "Unknown"
if pipeline is not None and hasattr(pipeline, "named_steps"):
    clf = pipeline.named_steps.get("clf", None)
    if clf is not None:
        model_name = clf.__class__.__name__

st.markdown(f"<h4 style='color:{PRIMARY}; margin-bottom:1.5rem;'>Model Used: <b>{model_name}</b></h4>", unsafe_allow_html=True)

# ----------------------
# Initialize Session State for History
# ----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------
# Input Box + Buttons
# ----------------------
col1, col2 = st.columns([4, 1])
with col1:
    text = st.text_area("✍️ Enter your text below:", placeholder="Type or paste your sentence here…", height=150)
with col2:
    st.write("")  # spacer
    reset = st.button("🔄 Reset", use_container_width=True)

predict_btn = st.button("🔍 Predict Sentiment", type="primary", use_container_width=True)

# ----------------------
# Sentiment Prediction Logic
# ----------------------
EMOJI = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
COLOR = {"positive": POS, "neutral": NEU, "negative": NEG}

def explain_confidence(p):
    labels = sorted(p.keys(), key=lambda k: p[k], reverse=True)
    top, second = labels[0], labels[1] if len(labels) > 1 else None
    margin = p[top] - (p[second] if second else 0.0)

    if p[top] >= 0.8:
        return f"The model is **very confident** this text is **{top}** — strong emotional language detected."
    elif p[top] >= 0.6:
        return f"The model leans **{top}**, but there are some mixed signals."
    elif margin <= 0.1:
        return f"The text has mixed cues — prediction uncertainty is high."
    else:
        return f"Prediction: **{top}** with moderate confidence."

def get_proba(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        return None

# Handle reset
if reset:
    st.session_state.history = []
    st.rerun()

# --- Sentiment Prediction ---
EMOJI = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}
COLOR = {"positive": POS, "neutral": NEU, "negative": NEG}

def explain_confidence(p):
    labels = sorted(p.keys(), key=lambda k: p[k], reverse=True)
    top, second = labels[0], labels[1] if len(labels) > 1 else None
    margin = p[top] - (p[second] if second else 0.0)
    if p[top] >= 0.8:
        return f"The model is **very confident** this text is **{top}** — strong emotional language detected."
    if p[top] >= 0.6:
        return f"The model leans **{top}**, but there are some mixed signals."
    if margin <= 0.1:
        return f"The text has mixed cues — prediction uncertainty is high."
    return f"Prediction: **{top}** with moderate confidence."

def get_proba(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        return None

def build_label_map(classes):
    if classes is None:
        return None
    # numeric 0/1/2 → neg/neu/pos (your training convention)
    try:
        as_int = [int(c) for c in classes]
        if as_int == [0, 1, 2]:
            return {0: "negative", 1: "neutral", 2: "positive"}
    except Exception:
        pass
    # string classes → use directly if they match our names
    lower = [str(c).lower() for c in classes]
    if {"negative", "neutral", "positive"}.issubset(set(lower)):
        return {c: str(c).lower() for c in classes}
    # fallback: alphabetical -> negative, neutral, positive
    ordered = sorted(classes, key=lambda x: str(x))
    names = ["negative", "neutral", "positive"]
    return {c: names[i] if i < 3 else str(c) for i, c in enumerate(ordered)}

if predict_btn and pipeline is not None and text.strip():
    try:
        X = [text]
        raw_pred = pipeline.predict(X)
        proba = get_proba(pipeline, X)

        classes = getattr(pipeline[-1], "classes_", None)
        label_map = build_label_map(list(classes)) if classes is not None else None

        pred = label_map.get(raw_pred[0], str(raw_pred[0])) if label_map else str(raw_pred[0])
        probs = (
            {label_map.get(c, str(c)): float(proba[0, i]) for i, c in enumerate(classes)}
            if (proba is not None and classes is not None and label_map is not None)
            else None
        )

        # Display main result
        st.markdown(
            f"""
            <div class="pill" style="background:{COLOR.get(pred,'#4F46E5')}22; border-color:{COLOR.get(pred,'#4F46E5')}55; color:white">
                {EMOJI.get(pred, '💡')} <b>{pred.title()}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence bars
        if probs is not None:
            st.markdown("<h5 style='text-align:center; margin-top:15px;'>Confidence per Sentiment</h5>", unsafe_allow_html=True)
            for label in ["positive", "neutral", "negative"]:
                val = float(probs.get(label, 0.0))
                pct = int(round(val * 100))
                st.markdown(
                    f"""
                    <div class="prob-row">
                        <div style="width:100px"><span class="pill" style="background:{COLOR[label]}22; border-color:{COLOR[label]}55">{EMOJI[label]} {label.title()}</span></div>
                        <div class="bar"><div class="fill" style="width:{pct}%; background:{COLOR[label]}"></div></div>
                        <div style="width:46px; text-align:right; font-variant-numeric:tabular-nums">{pct}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Explanation
        if probs is not None:
            st.markdown(f"<div class='explainer'>🧠 {explain_confidence(probs)}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='explainer'>🧠 Model does not provide probability estimates; showing label only.</div>", unsafe_allow_html=True)

        # Add to session history
        st.session_state.history.insert(0, {"text": text.strip(), "label": pred})

    except Exception as e:
        st.error(f"Prediction failed: {e}")



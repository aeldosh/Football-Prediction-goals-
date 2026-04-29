"""
Football Player Performance Predictor
======================================
Final Year Data Science Laboratory Project
Predicts player Goals using Gradient Boosting Regressor
trained on FBRef 2024-2025 dataset with 10-Fold Cross Validation.
"""

import streamlit as st
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Football Player Performance Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS for Professional Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* ── Header Card ── */
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .header-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { transform: translateX(-30%) translateY(-30%); }
        50% { transform: translateX(30%) translateY(30%); }
    }
    .header-card h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    .header-card p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* ── Model Badge ── */
    .model-badge {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .model-badge .label {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .model-badge .value {
        color: #667eea;
        font-size: 1rem;
        font-weight: 700;
        background: rgba(102, 126, 234, 0.1);
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.25);
    }

    /* ── Input Section ── */
    .input-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .section-title {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Streamlit Widget Overrides ── */
    .stNumberInput label, .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stNumberInput"] input {
        background: #0f172a !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    /* ── Predict Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5) !important;
    }

    /* ── Prediction Result Card ── */
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid rgba(16, 185, 129, 0.4);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15);
        animation: fadeInUp 0.5s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card .result-label {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .result-card .result-value {
        color: #10b981;
        font-size: 3.5rem;
        font-weight: 800;
        text-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
    }
    .result-card .result-unit {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        margin-top: 2.5rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .footer p {
        color: #475569;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .footer .highlight {
        color: #667eea;
        font-weight: 600;
    }

    /* ── Player Name Field ── */
    .player-name-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .player-name-section .section-title {
        margin-bottom: 0.8rem;
    }
    div[data-testid="stTextInput"] input {
        background: #0f172a !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.6rem 1rem !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }
    div[data-testid="stTextInput"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* ── Player Name in Result ── */
    .result-card .player-name-display {
        color: #667eea;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        letter-spacing: 0.5px;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load the Best Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the pre-trained Gradient Boosting Regressor model."""
    model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
    model = joblib.load(model_path)
    return model


model = load_model()


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>⚽ Football Player Performance Predictor</h1>
    <p>Powered by Machine Learning &mdash; FBRef 2024&ndash;2025 Dataset</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Information Badge
# ─────────────────────────────────────────────
st.markdown("""
<div class="model-badge">
    <span class="label">🏆 Best Model:</span>
    <span class="value">Gradient Boosting Regressor</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Player Name Section
# ─────────────────────────────────────────────
st.markdown('<div class="player-name-section"><div class="section-title">👤 Player Information</div>', unsafe_allow_html=True)
player_name = st.text_input(
    "⚽ Player Name",
    value="",
    placeholder="Enter Player Name",
    help="Name of the football player (for display purposes only)"
)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Input Features Section
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-title">📊 Enter Player Statistics</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "🎂 Age",
        min_value=15,
        max_value=45,
        value=25,
        step=1,
        help="Player's age in years"
    )
    matches_played = st.number_input(
        "🏟️ Matches Played",
        min_value=0,
        max_value=60,
        value=20,
        step=1,
        help="Total matches played in the season"
    )
    minutes = st.number_input(
        "⏱️ Minutes",
        min_value=0,
        max_value=5400,
        value=1500,
        step=10,
        help="Total minutes played"
    )

with col2:
    assists = st.number_input(
        "🎯 Assists",
        min_value=0,
        max_value=30,
        value=5,
        step=1,
        help="Total assists provided"
    )
    penalty_goals = st.number_input(
        "🥅 Penalty Goals Made",
        min_value=0,
        max_value=15,
        value=1,
        step=1,
        help="Goals scored from penalty kicks"
    )
    yellow_cards = st.number_input(
        "🟨 Yellow Cards",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
        help="Total yellow cards received"
    )

with col3:
    red_cards = st.number_input(
        "🟥 Red Cards",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
        help="Total red cards received"
    )
    progressive_carries = st.number_input(
        "🏃 Progressive Carries",
        min_value=0,
        max_value=200,
        value=30,
        step=1,
        help="Carries that move the ball towards the opponent's goal"
    )
    progressive_passes = st.number_input(
        "📤 Progressive Passes",
        min_value=0,
        max_value=400,
        value=50,
        step=1,
        help="Passes that move the ball towards the opponent's goal"
    )


# ─────────────────────────────────────────────
# Divider
# ─────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Predict Button & Result
# ─────────────────────────────────────────────
if st.button("⚡  Predict Goals", use_container_width=True):
    # Prepare input array in the exact feature order used during training
    input_features = np.array([[
        age,
        matches_played,
        minutes,
        assists,
        penalty_goals,
        yellow_cards,
        red_cards,
        progressive_carries,
        progressive_passes
    ]])

    # Make prediction
    prediction = model.predict(input_features)[0]
    predicted_goals = max(0, round(prediction, 2))  # Ensure non-negative
    display_name = player_name.strip() if player_name.strip() else "Unknown Player"

    # Display result
    st.markdown(f"""
    <div class="result-card">
        <div class="player-name-display">⚽ Player: {display_name}</div>
        <div class="result-label">Predicted Goals</div>
        <div class="result-value">{predicted_goals}</div>
        <div class="result-unit">Based on Gradient Boosting Regressor &bull; 10-Fold Cross Validation</div>
    </div>
    """, unsafe_allow_html=True)

    # Display feature summary in an expander
    with st.expander("📋 View Input Summary", expanded=False):
        summary_data = {
            "Feature": [
                "Age", "Matches Played", "Minutes", "Assists",
                "Penalty Goals Made", "Yellow Cards", "Red Cards",
                "Progressive Carries", "Progressive Passes"
            ],
            "Value": [
                age, matches_played, minutes, assists,
                penalty_goals, yellow_cards, red_cards,
                progressive_carries, progressive_passes
            ]
        }
        st.table(summary_data)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <p>🎓 <span class="highlight">Final Year Data Science Laboratory Project</span></p>
    <p>Football Player Performance Analysis using FBRef 2024&ndash;2025 Dataset</p>
</div>
""", unsafe_allow_html=True)

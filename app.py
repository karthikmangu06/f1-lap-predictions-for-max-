# -*- coding: utf-8 -*-
"""
app.py
======
Streamlit Web Interface - Max Verstappen Lap Time Simulator

Loads the pre-trained XGBoost pipeline (max_lap_predictor.pkl) and lets
users simulate Max Verstappen's lap times by adjusting race conditions.

Usage:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION (must be the very first Streamlit call)
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Max Verstappen Lap Time Simulator",
    page_icon="F1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# CUSTOM CSS - Premium Dark Theme
# ------------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Orbitron:wght@700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a0a00 50%, #0d0d0d 100%);
        color: #f0f0f0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c1c1c 0%, #111111 100%);
        border-right: 1px solid #3d1f00;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    .hero-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b00, #ff9d00, #ff6b00);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }
    @keyframes shimmer {
        from { background-position: 0% center; }
        to   { background-position: 200% center; }
    }
    .hero-subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 1rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(255,107,0,0.15), rgba(255,157,0,0.05));
        border: 1px solid rgba(255, 107, 0, 0.5);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 107, 0, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.2em;
        color: #ff9d00;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-time {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        color: #ffffff;
        text-shadow: 0 0 30px rgba(255, 107, 0, 0.8);
        line-height: 1;
    }
    .result-ms {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #888888;
        margin-top: 0.5rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        min-width: 120px;
    }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #ff9d00; }
    .metric-label {
        font-size: 0.75rem;
        color: #888888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .driver-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(30, 65, 185, 0.25);
        border: 1px solid #1e41b9;
        border-radius: 100px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #6eb8ff;
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.25em;
        color: #ff6b00;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255,107,0,0.3);
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(90deg, #ff6b00, #ff9d00) !important;
        color: #000000 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.05em !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(255, 107, 0, 0.4) !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(255, 107, 0, 0.6) !important;
    }
    div[data-testid="stButton"] > button:active { transform: translateY(0px) !important; }

    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #ff6b00 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------

# Only circuits the model was trained on should be listed here.
TRACK_LAPS = {
    "70th Anniversary": 52, "Abu Dhabi": 58, "Australia": 58,
    "Austria": 71, "Azerbaijan": 51, "Bahrain": 57,
    "Belgium": 44, "Brazil": 71, "Canada": 70,
    "China": 56, "Emilia Romagna": 63, "France": 53,
    "Germany": 64, "Great Britain": 52, "Hungary": 70,
    "Italy": 53, "Japan": 53, "Las Vegas": 50,
    "Malaysia": 56, "Mexico": 71, "Mexico City": 71,
    "Miami": 57, "Monaco": 78, "Netherlands": 72,
    "Qatar": 57, "Saudi Arabia": 50, "Sao Paulo": 71,
    "Spain": 66, "Styria": 71, "United States": 56,
}

FEATURE_COLUMNS = [
    "LapNumber", "Compound", "TyreLife", "TyreLife_Squared", "Track",
    "Stint", "FuelLapsRemaining", "IsPushing", "IsClearTrack",
]

TEST_CASES = [
    {"name": "1. Monza (Power-hungry)", "circuit": "Italy", "lap_number": 12,
     "stint_number": 1, "tyre_compound": "MEDIUM", "tyre_age": 11,
     "fuel_laps_remaining": 42, "driver_pushing": True, "clear_track": True},
    {"name": "2. Spa (Power-hungry + degradation)", "circuit": "Belgium", "lap_number": 28,
     "stint_number": 1, "tyre_compound": "HARD", "tyre_age": 22,
     "fuel_laps_remaining": 18, "driver_pushing": True, "clear_track": True},
    {"name": "3. Silverstone (Balanced high-speed)", "circuit": "Great Britain", "lap_number": 35,
     "stint_number": 2, "tyre_compound": "MEDIUM", "tyre_age": 14,
     "fuel_laps_remaining": 12, "driver_pushing": True, "clear_track": True},
    {"name": "4. Barcelona (Balanced technical)", "circuit": "Spain", "lap_number": 22,
     "stint_number": 1, "tyre_compound": "MEDIUM", "tyre_age": 21,
     "fuel_laps_remaining": 25, "driver_pushing": True, "clear_track": True},
    # Singapore was replaced: it is not in TRACK_LAPS, so the model has probably not seen it.
    {"name": "5. Hungaroring (Tyre-degradation heavy)", "circuit": "Hungary", "lap_number": 18,
     "stint_number": 1, "tyre_compound": "SOFT", "tyre_age": 8,
     "fuel_laps_remaining": 52, "driver_pushing": True, "clear_track": True},
    {"name": "6. Bahrain (Tyre monster, late stint)", "circuit": "Bahrain", "lap_number": 41,
     "stint_number": 2, "tyre_compound": "HARD", "tyre_age": 29,
     "fuel_laps_remaining": 8, "driver_pushing": True, "clear_track": True},
    {"name": "7. COTA (Bumpy + degradation)", "circuit": "United States", "lap_number": 31,
     "stint_number": 1, "tyre_compound": "MEDIUM", "tyre_age": 24,
     "fuel_laps_remaining": 15, "driver_pushing": True, "clear_track": True},
    {"name": "8. Interlagos (Late-race low-fuel)", "circuit": "Brazil", "lap_number": 55,
     "stint_number": 2, "tyre_compound": "MEDIUM", "tyre_age": 18,
     "fuel_laps_remaining": 4, "driver_pushing": True, "clear_track": True},
]

# ------------------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once)
# ------------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model ...")
def load_model(model_path: str = "max_lap_predictor.pkl"):
    """Load the trained Pipeline from disk, or return None if missing."""
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        return None


# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def ms_to_laptime(ms: float) -> str:
    """Convert milliseconds to the F1 format m:ss.mmm (e.g. 1:34.321)."""
    total_ms = int(round(ms))
    minutes, rem = divmod(total_ms, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def build_input_df(
    lap_number: int,
    compound: str,
    tyre_life: int,
    track: str,
    stint: int,
    fuel_laps_remaining: int,
    is_pushing: bool,
    is_clear_track: bool,
) -> pd.DataFrame:
    """Build a one-row DataFrame matching the training feature schema."""
    df = pd.DataFrame(
        {
            "LapNumber": [lap_number],
            "Compound": [compound.upper()],
            "TyreLife": [tyre_life],
            "TyreLife_Squared": [tyre_life ** 2],
            "Track": [track],
            "Stint": [stint],
            "FuelLapsRemaining": [fuel_laps_remaining],
            "IsPushing": [is_pushing],
            "IsClearTrack": [is_clear_track],
        }
    )
    return df[FEATURE_COLUMNS]


def run_all_tests(model) -> pd.DataFrame:
    """Run all batch test cases and return the results as a DataFrame."""
    results = []
    for test in TEST_CASES:
        input_df = build_input_df(
            lap_number=test["lap_number"],
            compound=test["tyre_compound"],
            tyre_life=test["tyre_age"],
            track=test["circuit"],
            stint=test["stint_number"],
            fuel_laps_remaining=test["fuel_laps_remaining"],
            is_pushing=test["driver_pushing"],
            is_clear_track=test["clear_track"],
        )
        try:
            predicted_ms = float(model.predict(input_df)[0])
            predicted_time_str = ms_to_laptime(predicted_ms)
            notes = "Success"
        except Exception as exc:
            predicted_time_str = "ERROR"
            notes = str(exc)

        results.append({
            "Test": test["name"],
            "Circuit": test["circuit"],
            "Lap": test["lap_number"],
            "Stint": test["stint_number"],
            "Compound": test["tyre_compound"],
            "Tyre Age": test["tyre_age"],
            "Fuel Left": test["fuel_laps_remaining"],
            "Predicted Lap Time": predicted_time_str,
            "Notes": notes,
        })
    return pd.DataFrame(results)


# ------------------------------------------------------------------------------
# MAIN LAYOUT
# ------------------------------------------------------------------------------

def main() -> None:
    model = load_model()

    st.markdown('<div class="hero-title">F1 Lap Time Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Powered by XGBoost &bull; FastF1 Telemetry Data</div>',
        unsafe_allow_html=True,
    )

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown(
            '<div class="driver-badge">#1 &nbsp;MAX VERSTAPPEN &nbsp;|&nbsp; Red Bull Racing</div>',
            unsafe_allow_html=True,
        )

    if model is None:
        st.error(
            "**Model file not found** (`max_lap_predictor.pkl`).\n\n"
            "Make sure the file is in the same folder as `app.py`."
        )
        st.stop()

    # --------------------------- SIDEBAR -------------------------------------
    with st.sidebar:
        st.markdown("## Race Conditions")
        st.markdown("---")

        st.markdown('<div class="section-header">CIRCUIT</div>', unsafe_allow_html=True)
        track_options = sorted(TRACK_LAPS.keys())
        track = st.selectbox(
            "Grand Prix Circuit",
            options=track_options,
            index=track_options.index("Bahrain"),
            help="Select the circuit. Race distance automatically updates the lap slider.",
        )

        st.markdown('<div class="section-header" style="margin-top:1.5rem">LAP &amp; TYRE</div>', unsafe_allow_html=True)

        max_laps = TRACK_LAPS.get(track, 70)
        lap_number = st.slider(
            "Lap Number",
            min_value=1,
            max_value=max_laps,
            value=min(25, max_laps),
            step=1,
            help=f"Which lap of the race is VER on? ({track} = {max_laps} laps)",
        )

        # Keyed by lap/track so the default re-syncs when either changes.
        fuel_laps_remaining = st.slider(
            "Fuel Laps Remaining",
            min_value=0,
            max_value=max_laps,
            value=max(0, max_laps - lap_number),
            step=1,
            key=f"fuel_{track}_{lap_number}",
            help="Laps left in the race, a proxy for current fuel load.",
        )

        compound = st.selectbox(
            "Tyre Compound",
            options=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
            index=1,
            help="The tyre compound currently fitted.",
        )

        colour_map = {
            "SOFT": "Red (S)", "MEDIUM": "Yellow (M)", "HARD": "White (H)",
            "INTERMEDIATE": "Green (I)", "WET": "Blue (W)",
        }
        st.markdown(f"**Selected:** {colour_map[compound]} compound")

        tyre_life = st.slider(
            "Tyre Age (laps)",
            min_value=1, max_value=50, value=10, step=1,
            help="How many laps have been completed on the current set of tyres?",
        )

        st.markdown('<div class="section-header" style="margin-top:1.5rem">STRATEGY</div>', unsafe_allow_html=True)

        stint = st.number_input(
            "Stint Number",
            min_value=1, max_value=5, value=1, step=1,
            help="Stint 1 = tyres since race start.",
        )

        st.markdown('<div class="section-header" style="margin-top:1.5rem">TRACK STATUS</div>', unsafe_allow_html=True)

        is_pushing = st.checkbox(
            "Driver is Pushing", value=True,
            help="Is the driver attacking / running at optimal pace?",
        )
        is_clear_track = st.checkbox(
            "Clear Track", value=True,
            help="Is the track clear of traffic, safety cars, or yellow flags?",
        )

        st.markdown("---")
        simulate_clicked = st.button("SIMULATE LAP", use_container_width=True)

        st.caption(
            "Model trained on FastF1 telemetry data. "
            "Predictions are estimates based on historical patterns."
        )

    # --------------------------- MAIN AREA -----------------------------------
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        if simulate_clicked:
            input_df = build_input_df(
                lap_number=lap_number,
                compound=compound,
                tyre_life=tyre_life,
                track=track,
                stint=int(stint),
                fuel_laps_remaining=fuel_laps_remaining,
                is_pushing=is_pushing,
                is_clear_track=is_clear_track,
            )

            try:
                with st.spinner("Calculating predicted lap time ..."):
                    predicted_ms = float(model.predict(input_df)[0])

                formatted_time = ms_to_laptime(predicted_ms)

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Lap Time</div>
                        <div class="result-time">{formatted_time}</div>
                        <div class="result-ms">{predicted_ms:,.1f} ms &nbsp;|&nbsp; {predicted_ms/1000:.3f} s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="metric-row">
                        <div class="metric-card"><div class="metric-value">{compound.title()}</div><div class="metric-label">Compound</div></div>
                        <div class="metric-card"><div class="metric-value">{tyre_life}</div><div class="metric-label">Tyre Age (laps)</div></div>
                        <div class="metric-card"><div class="metric-value">Lap {lap_number}</div><div class="metric-label">Race Lap</div></div>
                        <div class="metric-card"><div class="metric-value">Stint {int(stint)}</div><div class="metric-label">Stint</div></div>
                        <div class="metric-card"><div class="metric-value">{fuel_laps_remaining}</div><div class="metric-label">Fuel Laps Left</div></div>
                        <div class="metric-card"><div class="metric-value">{track}</div><div class="metric-label">Circuit</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                st.markdown("### Simulation Notes")
                tyre_state = (
                    "Fresh - expected near peak grip."
                    if tyre_life <= 5
                    else "Moderate degradation in effect."
                    if tyre_life <= 20
                    else "Significant wear - grip will be limited."
                )
                compound_note = {
                    "SOFT": "High grip, fast degradation.",
                    "MEDIUM": "Balanced performance.",
                    "HARD": "Low degradation, slower warm-up.",
                    "INTERMEDIATE": "Mixed-conditions tyre, optimal in damp/drying track.",
                    "WET": "Full wet tyre, maximum water displacement.",
                }.get(compound, "")
                status_note = (
                    "Optimal driving conditions."
                    if (is_pushing and is_clear_track)
                    else "Pace impacted by traffic/flags or tyre saving."
                )
                phase = (
                    "Early stint" if lap_number <= 20
                    else "Mid race" if lap_number <= 45
                    else "Final phase"
                )

                st.info(
                    f"**Tyre condition:** {tyre_state}  \n"
                    f"**Compound note:** {compound_note}  \n"
                    f"**Race phase:** {phase}.  \n"
                    f"**Pace status:** {status_note}"
                )

            except Exception as exc:
                st.error(
                    f"Prediction failed: `{exc}`\n\n"
                    "Ensure your model was trained with the same feature set."
                )
        else:
            st.markdown(
                """
                <div style="border: 1px dashed rgba(255,107,0,0.3); border-radius: 16px;
                            padding: 3rem 2rem; text-align: center; color: #666;">
                    <div style="font-size:3rem;margin-bottom:1rem">F1</div>
                    <div style="font-family:'Orbitron',sans-serif;font-size:1rem;color:#888;letter-spacing:0.1em">
                        CONFIGURE CONDITIONS IN THE SIDEBAR<br>THEN PRESS SIMULATE
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown("### How It Works")
        st.markdown(
            """
            This simulator uses a **machine learning model** trained on real
            F1 telemetry captured by the [FastF1](https://theoehrly.github.io/Fast-F1/)
            library.

            **ML Architecture:**
            - **Algorithm:** XGBoost Regressor (gradient-boosted trees)
            - **Encoding:** One-Hot Encoding for compound & circuit
            - **Pipeline:** `ColumnTransformer -> XGBRegressor`

            **Features used to predict lap time:**
            | Feature | Description |
            |---|---|
            | Lap Number | Race lap (circuit-specific max) |
            | Compound | Tyre type (S/M/H/I/W) |
            | Tyre Age | Laps on current set |
            | **Tyre Age²** | **Non-linear degradation signal** |
            | Track | Grand Prix circuit (environmental proxy) |
            | **Stint** | **Which pit-stop window** |
            | **Fuel Laps Remaining** | **Fuel load proxy (laps left)** |
            | **Is Pushing** | **Driver attack pace vs saving** |
            | **Is Clear Track** | **Absence of traffic / yellow flags** |

            **Data Source:** FastF1 official telemetry API
            """
        )

        st.markdown("---")
        st.markdown("### About #1")
        st.markdown(
            """
            **Max Verstappen** is a four-time consecutive Formula 1 World
            Champion (2021-2024) driving for Red Bull Racing. Known for his
            exceptional tyre management and consistency, his lap data provides
            a rich signal for machine learning.
            """
        )

    # --------------------------- BATCH TESTS ---------------------------------
    st.markdown("---")
    st.markdown("### Batch Simulations")
    with st.expander("Run Pre-Configured Test Cases"):
        if st.button("Execute Batch Tests", key="batch_btn", use_container_width=True):
            with st.spinner("Running batch simulations..."):
                # Saved in session_state so results survive reruns.
                st.session_state["batch_results"] = run_all_tests(model)

        if "batch_results" in st.session_state:
            st.dataframe(st.session_state["batch_results"], use_container_width=True)


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

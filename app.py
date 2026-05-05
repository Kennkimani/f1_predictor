import streamlit as st
import pandas as pd
import joblib
import os

# Page Config
st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

# Load Model
@st.cache_resource
def load_model():
    model_path = 'artifacts/f1_logistic_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

if model is None:
    st.error(" Model artifact not found. Please ensure 'artifacts/f1_logistic_model.pkl' is uploaded.")
else:
    st.title("🏎️ Full-Scale F1 Performance Predictor")
    st.markdown("Adjust all race parameters below to see the predicted outcome.")

    # Load driver list for the dropdown
    @st.cache_data
    def get_meta_data():
        df = pd.read_csv('f1_model_ready.csv')
        return sorted(df['Driver'].unique()), sorted(df['Team'].unique()), sorted(df['GrandPrix'].unique())

    drivers, teams, gps = get_meta_data()

    # --- UI Layout ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Driver & Team")
        selected_driver = st.selectbox("Driver", drivers)
        selected_team = st.selectbox("Team", teams)
        selected_gp = st.selectbox("Grand Prix", gps)
        year = st.selectbox("Season", [2024, 2025, 2026])

    with col2:
        st.subheader("⏱️ Race Pace")
        lap_time = st.number_input("Avg Lap Time (s)", 60.0, 120.0, 85.0)
        s1 = st.number_input("Best Sector 1", 10.0, 40.0, 22.0)
        s2 = st.number_input("Best Sector 2", 10.0, 50.0, 30.0)
        s3 = st.number_input("Best Sector 3", 10.0, 40.0, 23.0)
        tyre_life = st.slider("Tyre Life (Laps)", 0, 60, 10)

    with col3:
        st.subheader("📈 Form & Car")
        d_form = st.slider("Driver Form (Avg Pts)", 0.0, 25.0, 12.0)
        t_perf = st.slider("Team Performance", 0.0, 1000.0, 300.0)
        t_hist = st.slider("Track History (Avg Pts)", 0.0, 25.0, 10.0)
        speed = st.slider("Top Speed (km/h)", 280, 360, 325)
        rpm = st.slider("Avg RPM", 8000, 13000, 11000)

    # --- Prediction ---
    if st.button("Calculate Win Probability", use_container_width=True):
        # Create input row with the exact columns your model expects
        input_data = pd.DataFrame({
            'Year': [year],
            'GrandPrix': [selected_gp],
            'Driver': [selected_driver],
            'Team': [selected_team],
            'LapTime': [lap_time],
            'Sector1Time': [s1],
            'Sector2Time': [s2],
            'Sector3Time': [s3],
            'TyreLife': [tyre_life],
            'speed': [speed],
            'rpm': [rpm],
            'driver_form': [d_form],
            'team_performance': [t_perf],
            'track_history': [t_hist]
        })

        # Get results
        prob = model.predict_proba(input_data)[0][1]
        
        st.divider()
        st.write(f"### Predicted Win Probability for {selected_driver}:")
        st.header(f"{prob:.1%}")
        st.progress(prob)

        if prob > 0.5:
            st.balloons()
            st.success("Strong favorite to win!")
        elif prob > 0.15:
            st.info("Podium contender.")
        else:
            st.warning("Low probability of a win based on current features.")

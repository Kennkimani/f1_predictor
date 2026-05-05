import streamlit as st
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️")

st.title("🏎️ F1 Race Predictor")

# --- 1. Load Model with Error Handling ---
model_path = 'artifacts/f1_logistic_model.pkl'

if not os.path.exists(model_path):
    st.error(f"Model file not found at `{model_path}`. Did you run `model.py` first?")
else:
    model = joblib.load(model_path)
    st.success("Prediction Model Loaded")

    # --- 2. User Inputs ---
    st.sidebar.header("Race Conditions")
    grid = st.sidebar.slider("Starting Grid Position", 1, 20, 1)
    year = st.sidebar.selectbox("Season", [2024, 2025])
    gp = st.sidebar.text_input("Grand Prix Name (e.g. Spain)", "Austria")
    
    # --- 3. Prediction Button ---
    if st.button("Predict Outcome"):
        # Create input row matching your model's 16 columns
        # Fill secondary features with averages so it doesn't crash
        input_data = pd.DataFrame({
            'Year': [year],
            'GrandPrix': [gp],
            'Driver': ['VER'],  # Placeholder
            'Team': ['Red Bull Racing'], # Placeholder
            'GridPosition': [grid],
            'LapTime': [90.0],   # Placeholder average
            'Sector1Time': [22.0],
            'Sector2Time': [30.0],
            'Sector3Time': [23.0],
            'TyreLife': [10.0],
            'speed': [320.0],
            'rpm': [11000.0],
            'driver_form': [15.0],
            'team_performance': [40.0],
            'track_history': [12.0]
        })

        # Get Prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # --- 4. Display Results ---
        st.divider()
        if prediction == 1:
            st.balloons()
            st.header(f"Predicted Winner!")
        else:
            st.header("🏁 Predicted Finish: Outside P1")
        
        st.write(f"Confidence Level: **{prob:.1%}**")


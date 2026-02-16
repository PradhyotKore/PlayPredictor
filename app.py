import streamlit as st
import pandas as pd
import joblib
import subprocess
import os
import time

st.set_page_config(page_title="Football AI Predictor", layout="centered")

@st.cache_resource
def load_models(team_name=None):
    if team_name:
        safe_name = "".join([c for c in team_name if c.isalnum() or c==' ']).rstrip()
        filename = f'football_model_{safe_name}.pkl'
        if os.path.exists(filename):
            try:
                return joblib.load(filename), True 
            except:
                pass
    
    try:
        return joblib.load('football_model.pkl'), False 
    except FileNotFoundError:
        return None, False

def load_team_stats():
    try:
        df = pd.read_csv('team_stats.csv')
        return dict(zip(df['Offense'], df['team_pass_rate']))
    except:
        return None

st.title("Football Play Predictor")
st.markdown("AI-powered play calling predictions based on NCAA data.")

team_stats = load_team_stats()
models, _ = load_models() 

tab1, tab2 = st.tabs(["Predictor", "Data Manager"])

with tab1:
    if models is None:
        st.error("Model file not found! Please go to Data Manager and train the model.")
    else:
        st.subheader("Game Situation")
        
        col1, col2 = st.columns(2)
        with col1:
            down = st.number_input("Down", 1, 4, 1)
            distance = st.number_input("Distance", 1, 99, 10)
            yard_line = st.number_input("Yards to Goal", 1, 99, 75)
            st.divider()
            
            st.subheader("Opponent Tendency")
            if team_stats:
                team_names = sorted(list(team_stats.keys()))
                selected_team = st.selectbox("Select Specific Opponent (Optional)", ["Generic (No Weighting)"] + team_names)
                
                if selected_team != "Generic (No Weighting)":
                    default_pass = float(team_stats[selected_team])
                    
                    models, is_custom = load_models(selected_team)
                    if is_custom:
                        st.success(f"Using Custom Model specialized for {selected_team}!")
                    else:
                        st.info(f"Using Generic Model with {selected_team} stats. (Go to Data Manager -> Train Custom Model for better results)")
                else:
                    default_pass = 0.50
                    models, _ = load_models(None)
            
            else:
                default_pass = 0.50
                st.warning("No team stats found. Train model first.")
                models, _ = load_models(None)

            if models is None:
                 st.error("Model file not found! Please train the model.")
                 st.stop()

            pass_rate = st.slider("Pass Rate (Auto-filled)", 0.0, 1.0, default_pass, 0.01, 
                                  help="0.2 = Run Heavy, 0.8 = Air Raid")
        
        with col2:
            score_diff = st.number_input("Score Diff (You - Them)", -100, 100, 0, help="Negative means you are losing.")
            
            st.write("Time Remaining (HS 12m Quarters)")
            c_q, c_m, c_s = st.columns(3)
            with c_q:
                quarter = st.number_input("Qtr", 1, 4, 1)
            with c_m:
                minutes = st.number_input("Min", 0, 12, 12)
            with c_s:
                seconds = st.number_input("Sec", 0, 59, 0)
            
            hs_seconds_left = ((4 - quarter) * 12 * 60) + (minutes * 60) + seconds
            model_seconds = hs_seconds_left * 1.25
            
            st.caption(f"Scaled to NCAA Time: {int(model_seconds)} sec remaining")

            formation = st.selectbox("Offensive Formation", 
                                     ["Standard", "Shotgun", "Pistol", "Trips", "Empty", "Bunch", "Wildcat"])

        if st.button("Predict Play", type="primary"):
            input_data = pd.DataFrame({
                'down': [down],
                'distance': [distance],
                'yard_line': [yard_line],
                'score_diff': [score_diff],
                'seconds_remaining': [model_seconds],
                'team_pass_rate': [pass_rate],
                'offensive_formation': [formation]
            })
            
            play_type = models['play_type'].predict(input_data)[0]
            cover_scheme = models['cover_scheme'].predict(input_data)[0]
            
            probs = models['play_type'].predict_proba(input_data)[0]
            classes = models['play_type'].classes_
            
            run_prob = probs[list(classes).index('Run')]
            pass_prob = probs[list(classes).index('Pass')]
            
            st.divider()
            r_col, l_col = st.columns(2)
            with r_col:
                st.metric("Predicted Play Type", play_type)
                if play_type == 'Run':
                    st.progress(run_prob, text=f"Confidence: {run_prob:.0%}")
                else:
                    st.progress(pass_prob, text=f"Confidence: {pass_prob:.0%}")
            
            with l_col:
                st.metric("Recommended Defense", cover_scheme)
                st.caption("Based on Down/Distance data")
            
            if models['play_direction']:
                direction = models['play_direction'].predict(input_data)[0]
                st.info(f"Predicted Direction: **{direction}** (Low Confidence)")

with tab2:
    st.header("Data Management")
    st.markdown("Use this tab to update data or train specialized models.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("1. General Update")
        if st.button("Process Data & Retrain Generic Model"):
            with st.status("Processing...", expanded=True) as status:
                st.write("Running Data Loader (Parsing CSV)...")
                try:
                    subprocess.run(["python", "ncaa_data_loader.py"], check=True)
                    st.write("Data Loaded.")
                except subprocess.CalledProcessError:
                    st.error("Failed to load data.")
                    st.stop()
                
                st.write("Training Generic Model...")
                try:
                    subprocess.run(["python", "football_predictor.py", "--test-only"], check=True)
                    st.write("Models Trained & Saved.")
                except subprocess.CalledProcessError:
                    st.error("Training failed.")
                    st.stop()
                
                status.update(label="Complete!", state="complete", expanded=False)
                st.success("System updated!")
                st.cache_resource.clear()

    with col_b:
        st.subheader("2. Train Custom Team Model")
        st.markdown("Creating a specialized model for a specific opponent by weighting their data 10x higher.")
        
        if team_stats:
            team_names = sorted(list(team_stats.keys()))
            target_team = st.selectbox("Select Team to Train For", team_names)
            
            if st.button(f"Train Model for {target_team}"):
                with st.status(f"Training Custom Model for {target_team}...", expanded=True) as status:
                    try:
                        subprocess.run(["python", "football_predictor.py", "--test-only", "--target_team", target_team], check=True)
                        st.write(f"Trained & Saved `football_model_{target_team}.pkl`")
                    except subprocess.CalledProcessError:
                        st.error("Training failed.")
                        st.stop()
                    
                    status.update(label="Complete!", state="complete", expanded=False)
                    st.success(f"Custom Model for {target_team} ready! Select them in the Predictor tab.")
                    st.cache_resource.clear()
        else:
            st.warning("Load data first.")


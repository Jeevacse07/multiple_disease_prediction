import streamlit as st
import joblib
import numpy as np

# Load models
parkinson_model = joblib.load('parkinson_xgboost_model.pkl')
kidney_model = joblib.load('kidney_model_xgboost_model.pkl')
liver_model = joblib.load('liver_xgboost_model.pkl')

# Title
st.title("Multi-Disease Prediction App")

# Define feature keys for each disease
parkinson_features = ['p_spread1', 'p_PPE', 'p_DFA', 'p_RPDE', 'p_spread2',
                      'p_HNR', 'p_NHR', 'p_MDVP_Fo_Hz', 'p_MDVP_Shimmer', 'p_MDVP_Jitter']
kidney_features = ['k_hemo', 'k_sg', 'k_pcv', 'k_pc', 'k_sod',
                   'k_pot', 'k_ba', 'k_age', 'k_cad', 'k_pcc']
liver_features = ['l_age', 'l_gender', 'l_total_bilirubin', 'l_alk_phos',
                  'l_ast', 'l_albumin', 'l_ag_ratio']

# Tabs
tabs = st.tabs(["Parkinson's", "Kidney", "Liver"])

### TAB 1: Parkinson's
with tabs[0]:
    st.subheader("Parkinson's Disease Prediction")

    # Initialize state
    for key in parkinson_features:
        st.session_state.setdefault(key, 0.0)

    # Reset button BEFORE inputs
    if st.button("Reset Parkinson's Form"):
        for key in parkinson_features:
            st.session_state[key] = 0.0
        st.rerun()

    # Inputs
    parkinson_values = []
    for key in parkinson_features:
        val = st.number_input(key, key=key, value=st.session_state[key])
        parkinson_values.append(val)

    if st.button("Predict Parkinson's"):
        input_array = np.array(parkinson_values).reshape(1, -1)
        result = parkinson_model.predict(input_array)[0]
        st.success("Disease Present" if result == 1 else "No Disease")

### TAB 2: Kidney
with tabs[1]:
    st.subheader("Kidney Disease Prediction")

    for key in kidney_features:
        st.session_state.setdefault(key, 0.0)

    if st.button("Reset Kidney Form"):
        for key in kidney_features:
            st.session_state[key] = 0.0
        st.rerun()

    kidney_values = []
    for key in kidney_features:
        val = st.number_input(key, key=key, value=st.session_state[key])
        kidney_values.append(val)

    if st.button("Predict Kidney Disease"):
        input_array = np.array(kidney_values).reshape(1, -1)
        result = kidney_model.predict(input_array)[0]
        st.success("Disease Present" if result == 1 else "No Disease")

### TAB 3: Liver
with tabs[2]:
    st.subheader("Liver Disease Prediction")

    for key in liver_features:
        st.session_state.setdefault(key, 0.0)

    if st.button("Reset Liver Form"):
        for key in liver_features:
            st.session_state[key] = 0.0
        st.rerun()

    liver_values = []
    for key in liver_features:
        val = st.number_input(key, key=key, value=st.session_state[key])
        liver_values.append(val)

    if st.button("Predict Liver Disease"):
        input_array = np.array(liver_values).reshape(1, -1)
        result = liver_model.predict(input_array)[0]
        st.success("Disease Present" if result == 1 else "No Disease")

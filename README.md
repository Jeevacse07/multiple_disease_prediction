# Multi-Disease Prediction App

This Streamlit-based web application allows users to predict the presence of Parkinson's Disease, Kidney Disease, and Liver Disease using pre-trained XGBoost models. The models are trained on datasets with various medical features and provide predictions based on user input.

## Project Overview

The purpose of this project is to create a tool that can predict the likelihood of three different diseases based on user input. The application supports the following diseases:

1. **Parkinson's Disease**
2. **Kidney Disease**
3. **Liver Disease**

### Features

- **Parkinson's Disease Prediction**: Predicts whether a person has Parkinson's Disease based on voice-related features.
- **Kidney Disease Prediction**: Predicts the likelihood of kidney disease using medical features like blood pressure, age, and lab test results.
- **Liver Disease Prediction**: Predicts whether a person has liver disease using clinical test results and health metrics.

## Technologies Used

- **Python**: For model training and app development.
- **Streamlit**: For creating the interactive web interface.
- **XGBoost**: Machine learning algorithm for building predictive models for each disease.
- **Joblib**: For saving and loading pre-trained models.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/multi-disease-prediction.git
   ```

2. **Download the trained models**:
   Ensure the following `.pkl` model files are in the project directory:
   - `parkinson_xgboost_model.pkl`
   - `kidney_model_xgboost_model.pkl`
   - `liver_xgboost_model.pkl`

   These models were trained using their respective disease datasets.

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run disease_predict_app.py
   ```

## How to Use the App

1. **Select a disease tab**: 
   There are three tabs for disease prediction: **Parkinson's**, **Kidney**, and **Liver**. Each tab corresponds to the prediction model for that disease.
   
2. **Input the necessary features**: 
   Each tab presents a series of numeric inputs specific to the disease model. Enter the values for the respective features.

3. **Click on "Predict"**: 
   After entering the required values, click the **Predict** button to receive a prediction on whether the disease is present or not.

4. **Reset Form**: 
   To clear the form and start over, click the **Reset** button.

## Model Information

### Parkinson's Disease Model
- **Model Type**: XGBoost
- **Features Used**:
  - `p_spread1`, `p_PPE`, `p_DFA`, `p_RPDE`, `p_spread2`, `p_HNR`, `p_NHR`, `p_MDVP_Fo_Hz`, `p_MDVP_Shimmer`, `p_MDVP_Jitter`

### Kidney Disease Model
- **Model Type**: XGBoost
- **Features Used**:
  - `k_hemo`, `k_sg`, `k_pcv`, `k_pc`, `k_sod`, `k_pot`, `k_ba`, `k_age`, `k_cad`, `k_pcc`

### Liver Disease Model
- **Model Type**: XGBoost
- **Features Used**:
  - `l_age`, `l_gender`, `l_total_bilirubin`, `l_alk_phos`, `l_ast`, `l_albumin`, `l_ag_ratio`

## Sample Input Files

Each disease tab has a sample CSV file for testing the app with example data. The structure of the sample data corresponds to the features used for each model.

- **Parkinson's Sample Data**: `parkinson_sample.csv`
- **Kidney Disease Sample Data**: `kidney_sample.csv`
- **Liver Disease Sample Data**: `liver_sample.csv`

## EDA Files

The preprocessing, exploratory data analysis (EDA), feature engineering, and model training steps for each disease are contained in the following Jupyter notebooks. These files provide detailed insights into how the models were developed:

- `parkinson_eda.ipynb`
- `kidney_eda.ipynb`
- `liver_eda.ipynb`

The final models have been saved as `.pkl` files for use in the app.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

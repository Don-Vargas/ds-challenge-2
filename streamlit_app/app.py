import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit app configuration ---
st.set_page_config(page_title="Concept Drift Simulator", layout="wide")
st.title("Concept Drift Simulation Visualizer")

# --- User Inputs ---
st.sidebar.header("Simulation Settings")
file_path = st.sidebar.text_input(
    "CSV File Path", "src/research/dataset/test/ds4.csv"
)
features = st.sidebar.text_area(
    "Features (comma-separated)", "feature1,feature2"
).split(",")
features = [f.strip() for f in features if f.strip()]
target = st.sidebar.text_input("Target column", "target")
alpha = st.sidebar.slider("Alpha (significance level)", 0.01, 0.1, 0.05)

simulate_button = st.sidebar.button("Run Simulation")

# --- Run simulation via FastAPI ---
if simulate_button:
    if not file_path or not features or not target:
        st.error("Please provide a valid file path, features, and target column.")
    else:
        st.info("Running concept drift simulation...")

        payload = {
            "file_path": file_path,
            "features": features,
            "target": target,
            "alpha": alpha
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/simulate-drift", json=payload
            )
            response.raise_for_status()
            result = response.json()

            # --- Load dataframes ---
            reference_df = pd.DataFrame(result["reference_data"])
            drifted_df = pd.DataFrame(result["drifted_data"])
            drift_results = pd.DataFrame(result["drift_results"])

            st.success("Simulation completed!")

            # --- Show drift results ---
            st.subheader("Drift Detection Results")
            st.dataframe(drift_results)

            # --- Plot distributions ---
            st.subheader("Feature Distributions Before vs After Drift")
            for feature in features:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

                sns.countplot(x=feature, data=reference_df, ax=ax[0])
                ax[0].set_title(f"{feature} - Reference")

                sns.countplot(x=feature, data=drifted_df, ax=ax[1])
                ax[1].set_title(f"{feature} - Drifted")

                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error calling API: {e}")

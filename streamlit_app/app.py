import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Concept Drift Simulator", layout="wide")

# Title
st.title("Concept Drift Simulator")

# -----------------------------
# Sidebar for settings
# -----------------------------
st.sidebar.header("Settings")

file_path = st.sidebar.text_input("Enter dataset CSV file path:")
simulate_button = st.sidebar.button("Simulate Drift Step-by-Step")

# Optional: Add more settings
st.sidebar.markdown("---")
st.sidebar.markdown("Adjust visualization settings here.")
bar_alpha = st.sidebar.slider("Bar opacity", min_value=0.1, max_value=1.0, value=0.4)
max_rows = st.sidebar.number_input("Max rows to preview", min_value=10, max_value=1000, value=50, step=10)

# -----------------------------
# Main app logic
# -----------------------------
if simulate_button:
    if not file_path:
        st.error("Please enter a file path.")
    else:
        try:
            api_url = "http://127.0.0.1:8000/simulate-drift-file"
            response = requests.post(api_url, params={"file_path": file_path})
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                st.error(f"Error: {data.get('message')}")
            else:
                drifted_data = pd.DataFrame(data["drifted_data"])
                feature_drift = data.get("feature_drift", {})
                target_drift = data.get("target_drift", {})

                # Step 1: Drifted dataset
                st.subheader("Step 1: Original vs Drifted Dataset Preview")
                st.markdown("Original Data (first few rows):")
                st.dataframe(pd.DataFrame(data["df"]).head(max_rows))
                st.markdown("Drifted Data (first few rows):")
                st.dataframe(drifted_data.head(max_rows))

                # Step 2: Feature distributions overlap
                st.subheader("Step 2: Feature Distributions Comparison")
                for feature in drifted_data.columns:
                    if feature == "target":
                        continue  # skip target for now
                    
                    st.markdown(f"**Feature: {feature}**")
                    
                    df_original = pd.DataFrame(data["df"])
                    pre_probs = df_original[feature].value_counts(normalize=True)
                    post_probs = drifted_data[feature].value_counts(normalize=True)
                    
                    categories = list(set(pre_probs.index).union(post_probs.index))
                    pre_probs = np.array([pre_probs.get(cat, 0) for cat in categories])
                    post_probs = np.array([post_probs.get(cat, 0) for cat in categories])
                    
                    fig, ax = plt.subplots(figsize=(6, 3))
                    x = np.arange(len(categories))
                    width = 0.35
                    ax.bar(x - width/2, pre_probs, width, label="Original", color="blue", alpha=bar_alpha)
                    ax.bar(x + width/2, post_probs, width, label="Drifted", color="red", alpha=bar_alpha)
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories, rotation=45, ha="right")
                    ax.set_ylabel("Probability")
                    ax.set_ylim(0, max(max(pre_probs), max(post_probs)) * 1.1)
                    ax.legend()
                    st.pyplot(fig)

                # Step 3: Target probabilities per feature
                st.subheader("Step 3: Target Probabilities per Feature")
                for feature in drifted_data.columns:
                    if feature == "target":
                        continue
                    
                    st.markdown(f"**Feature: {feature}**")
                    df_original = pd.DataFrame(data["df"])
                    
                    pre_target_probs = df_original.groupby(feature)["target"].mean()
                    post_target_probs = drifted_data.groupby(feature)["target"].mean()
                    
                    categories = list(set(pre_target_probs.index).union(post_target_probs.index))
                    pre_probs = np.array([pre_target_probs.get(cat, 0) for cat in categories])
                    post_probs = np.array([post_target_probs.get(cat, 0) for cat in categories])
                    
                    fig, ax = plt.subplots(figsize=(6, 3))
                    x = np.arange(len(categories))
                    width = 0.35
                    ax.bar(x - width/2, pre_probs, width, label="Original", color="green", alpha=bar_alpha)
                    ax.bar(x + width/2, post_probs, width, label="Drifted", color="orange", alpha=bar_alpha)
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories, rotation=45, ha="right")
                    ax.set_ylabel("Target Probability")
                    ax.set_ylim(0, 1)
                    ax.legend()
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Unexpected error: {e}")

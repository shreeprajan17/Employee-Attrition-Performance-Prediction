import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import data_loader
import classification
import regression
import clustering

# --- PAGE SETUP ---
st.set_page_config(page_title="Employee Performance App", layout="wide")
st.title("📊 Employee Attrition and Performance Prediction")

# Load Data once
df = data_loader.load_dataset()

# --- SIDEBAR ---
menu = st.sidebar.radio("Modules", ["1. Classification", "2. Regression", "3. Clustering"])

# --- PAGE 1: CLASSIFICATION ---
if menu == "1. Classification":
    st.header("Retain: Predict Attrition")
    if st.button("Run Classification Model"):
        with st.spinner("Training..."):
            acc, report, cm = classification.run_classification(df)
            
            st.success(f"Accuracy: {acc:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Report")
                st.dataframe(pd.DataFrame(report).transpose())

# --- PAGE 2: REGRESSION ---
elif menu == "2. Regression":
    st.header("Reward: Predict Salary")
    if st.button("Run Salary Prediction"):
        with st.spinner("Comparing Models..."):
            results_df, y_test, preds = regression.run_regression(df)
            
            st.table(results_df)
            
            st.subheader("Actual vs Predicted (Random Forest)")
            fig, ax = plt.subplots()
            plt.scatter(y_test, preds['Random Forest'], alpha=0.5, color='purple')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
            st.pyplot(fig)

# --- PAGE 3: CLUSTERING ---
elif menu == "3. Clustering":
    st.header("Segment: Employee Profiling")
    if st.button("Generate Clusters"):
        fig_3d, profiles = clustering.run_clustering(df)
        st.plotly_chart(fig_3d, use_container_width=True)
        st.write("### Cluster Profiles")
        st.table(profiles)
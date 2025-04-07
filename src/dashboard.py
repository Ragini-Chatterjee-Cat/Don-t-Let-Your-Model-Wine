# src/dashboard.py
import streamlit as st
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix

st.title("Wine Quality Model Results 🍷")

results_path = st.text_input("Path to results folder", "results/wine_model")

try:
    with open(f"{results_path}/metrics.json") as f:
        metrics = json.load(f)

    st.subheader(" Accuracy")
    st.write(metrics["accuracy"])

    st.subheader(" Classification Report")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df)

    if st.checkbox("Show confusion matrix"):
        with open(f"{results_path}/model.pkl", "rb") as f:
            model = pickle.load(f)

        df = pd.read_csv("data/WineQT.csv")
        X = df.drop(columns=["quality", "Id"])
        y = df["quality"]
        y_pred = model.predict(X)

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

except Exception as e:
    st.error(f"Something went wrong: {e}")

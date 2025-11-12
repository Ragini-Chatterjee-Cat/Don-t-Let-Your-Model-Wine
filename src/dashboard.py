import streamlit as st
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix

st.title("Wine Quality Prediction Results ")

# I'm gettin the results path frm the txt input, so I know where my results r stored
results_path = st.text_input("Path to results folder", "results/wine_model")

try:
    # I am opening the metrics json file to get my model's accuracy and report
    with open(f"{results_path}/metrics.json") as f:
        metrics = json.load(f)

    # I show the model's accurecy here
    st.subheader(" Accuracy")
    st.write(metrics["accuracy"])

    # I also display the classifcation report to see more details
    st.subheader(" Classification Report")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df)

    # If I want to see the confusoin matrix, I can click this box
    if st.checkbox("Show confusion matrix"):
        with open(f"{results_path}/model.pkl", "rb") as f:
            model = pickle.load(f)

        # I load the dataset to compute predictions vs actual values
        df = pd.read_csv("data/WineQT.csv")
        X = df.drop(columns=["quality", "Id"])
        y = df["quality"]
        y_pred = model.predict(X)

        # Now I compute the confusion matrix
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        # I use seaborn's heatmap to plot the confusoin matrix :) 
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

except Exception as e:
    # If somethin goes wrong, I'll show the error message here :( 
    st.error(f"Somethin went wrong: {e}")

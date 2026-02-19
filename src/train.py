import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import mlflow
import mlflow.sklearn


def train_model(data_path):
    # Reading the data from the CSV from Kaggle
    df = pd.read_csv(data_path)

    # I drop the 'quality' and 'Id' columns so they don't get used as inputs
    X = df.drop(columns=['quality', 'Id'])
    y = df['quality']

    # I usually do an 80/20 split, it's a common practice
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # I'm using a RandomForestClassifier cause it's easy and performs decently well
    model = RandomForestClassifier(random_state=42)

    with mlflow.start_run():
        # Log training parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)

        model.fit(X_train, y_train)

        # Now I'll predict on the test set
        y_pred = model.predict(X_test)

        # I measure accuracy as a quick check, but I also get a classification report
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        for label, vals in report.items():
            if isinstance(vals, dict):
                label_clean = str(label).replace(" ", "_")
                for metric_name, val in vals.items():
                    if metric_name != "support":
                        mlflow.log_metric(f"{label_clean}_{metric_name}", val)

        # Log the model artifact
        mlflow.sklearn.log_model(model, "wine_quality_model")

    # I return the model, the metrics, and the actual and predicted values for later use
    return model, {'accuracy': acc, 'report': report}, y_test.tolist(), y_pred.tolist()

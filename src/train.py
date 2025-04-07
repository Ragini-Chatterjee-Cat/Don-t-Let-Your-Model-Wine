### src/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['quality', 'Id'])
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, {'accuracy': acc, 'report': report}, y_test.tolist(), y_pred.tolist()

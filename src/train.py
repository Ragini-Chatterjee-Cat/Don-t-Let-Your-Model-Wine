import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_model(data_path):
    # Reading the data from the CSV from kaggle 
    df = pd.read_csv(data_path)

    # I drop the 'quality' and 'Id' columns so they don't get used as inputs
    X = df.drop(columns=['quality', 'Id'])
    y = df['quality']

    # I usualy do an 80/20 split, it's a common practice  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # I'm using a RandomForestClassifier cause it's easy and performs decently well
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Now I'll predict on the test set
    y_pred = model.predict(X_test)

    # I measure accuracy as a quick check, but I also get a classification report
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True) 

    # I return the model, the metrics, and the actual and predicted values for later use
    return model, {'accuracy': acc, 'report': report}, y_test.tolist(), y_pred.tolist()

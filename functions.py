# functions.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def train_model(X_train, y_train, **kwargs):
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_pred  = model.predict(X_test)
    proba   = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy':    accuracy_score(y_test, y_pred),
        'precision':   precision_score(y_test, y_pred),
        'recall':      recall_score(y_test, y_pred),
        'f1_score':    f1_score(y_test, y_pred),
        'roc_auc':     roc_auc_score(y_test, proba)
    }

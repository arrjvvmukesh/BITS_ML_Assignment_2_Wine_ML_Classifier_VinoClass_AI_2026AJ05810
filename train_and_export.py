import os
import io
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def download_and_load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"[1/5] Fetching dataset from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), sep=";")
    
    # Defining binary target: 1 if quality >= 7 (High quality), else 0
    # Meets the min 12 attribute / >500 rows requirement
    df["target"] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns (11 features + 1 target).")
    return df

def main():
    os.makedirs("model", exist_ok=True)
    df = download_and_load_data()
    
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]
    
    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Save test_data.csv with features and target for app upload and repo submission
    test_df = X_test.copy()
    test_df["target"] = y_test
    test_df.to_csv("test_data.csv", index=False)
    print(f"[2/5] Saved test_data.csv with {len(test_df)} samples.")
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "model/preprocessor.joblib")
    
    # 5 Mandated Classification Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=42, kernel="rbf")
    }
    
    file_keys = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "KNN": "knn",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm"
    }
    
    print("[3/5] Training models and computing metrics...")
    results = []
    for name, model in models.items():
        # Training
        model.fit(X_train_scaled, y_train)
        
        # Prediction
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 6 Mandatory Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })
        
        # Persist model artifact
        save_path = os.path.join("model", f"{file_keys[name]}.joblib")
        joblib.dump(model, save_path)
    
    print("[4/5] Saved all 5 model artifacts inside model/ folder.")
    print("[5/5] Performance Summary Table:")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
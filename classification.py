import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def run_classification(df):
    # 1. Prepare Data
    df_cls = df.copy()
    df_cls['Attrition'] = df_cls['Attrition'].map({'Yes': 1, 'No': 0})
    df_encoded = pd.get_dummies(df_cls, drop_first=True)
    
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    
    # RFE & Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(rf, n_features_to_select=20, step=1)
    selector = selector.fit(X_train_res, y_train_res)
    
    X_train_rfe = selector.transform(X_train_res)
    X_test_rfe = selector.transform(X_test_scaled)
    
    rf_final = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_final.fit(X_train_rfe, y_train_res)
    
    # Prediction with Threshold
    y_probs = rf_final.predict_proba(X_test_rfe)[:, 1]
    y_pred = (y_probs >= 0.30).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, report, cm
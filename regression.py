import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def run_regression(df):
    df_reg = df.copy()
    
    # Custom Encoding
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
    df_reg['BusinessTravel'] = df_reg['BusinessTravel'].map(travel_map)
    df_reg = df_reg.drop('Attrition', axis=1)
    df_final = pd.get_dummies(df_reg, drop_first=True)
    
    X = df_final.drop('MonthlyIncome', axis=1)
    y = df_final['MonthlyIncome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Compare Models
    results = []
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        predictions[name] = y_pred
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append({"Model": name, "R2 Score": r2, "RMSE": rmse})
        
    return pd.DataFrame(results), y_test, predictions
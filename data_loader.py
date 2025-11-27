import pandas as pd

def load_dataset():
    df = pd.read_csv('Attrition.csv')
    
    # Basic Cleaning for everyone
    df.drop(['Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True)
    
    return df
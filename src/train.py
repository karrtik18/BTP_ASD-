 

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
import os  

warnings.filterwarnings('ignore')

 
url = 'https://raw.githubusercontent.com/karrtik18/BTP_ASD-/main/autism_dataset.csv'
df = pd.read_csv(url)
df.replace('?', np.nan, inplace=True)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age'].fillna(df['age'].median(), inplace=True)
categorical_cols_to_impute = ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'relation']
for col in categorical_cols_to_impute:
    df[col].fillna(df[col].mode()[0], inplace=True)

cols_to_drop = ['ethnicity', 'contry_of_res', 'age_desc', 'used_app_before', 'result']
df.drop(cols_to_drop, axis=1, inplace=True)

binary_map = {'yes': 1, 'no': 0, 'YES': 1, 'NO': 0}
df['jundice'] = df['jundice'].map(binary_map)
df['austim'] = df['austim'].map(binary_map)
df['Class/ASD'] = df['Class/ASD'].map(binary_map)
df = pd.get_dummies(df, columns=['gender', 'relation'], drop_first=True)

 
X = df.drop(['Class/ASD'], axis=1)
y = df['Class/ASD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train.values, y_train.values)

print("Model trained successfully.")

 
model_dir = '../models'
model_filename = 'xgboost_model.joblib'
model_path = os.path.join(model_dir, model_filename)

 
os.makedirs(model_dir, exist_ok=True)  

 
joblib.dump(model, model_path)  

print(f"Model saved to {model_path}")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("insurance.csv")

# ---------- PREPROCESSING ----------
df.drop_duplicates(inplace=True)

# Encoding
df['sex'] = df['sex'].map({"male": 0, "female": 1})
df['smoker'] = df['smoker'].map({"no": 0, "yes": 1})

df.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)

# One hot encoding
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# BMI category
df['bmi_category'] = pd.cut(
    df['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['Underweight','Normal','Overweight','obesity']
)

df = pd.get_dummies(df, columns=['bmi_category'], drop_first=True)

# Final features (same as your notebook)
final_df = df[['age', 'is_female', 'bmi', 'children', 'is_smoker',
               'region_southeast', 'bmi_category_obesity', 'charges']]

# ---------- SCALING ----------
scaler = StandardScaler()
cols = ['age', 'bmi', 'children']
final_df[cols] = scaler.fit_transform(final_df[cols])

# ---------- MODEL ----------
X = final_df.drop('charges', axis=1)
y = final_df['charges']

model = LinearRegression()
model.fit(X, y)

# ---------- SAVE ----------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved successfully!")
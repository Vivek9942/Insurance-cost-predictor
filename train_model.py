import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# ---------------- LOAD DATA ----------------
df = pd.read_csv("insurance.csv")

# ---------------- CLEANING ----------------
df = df.drop_duplicates().copy()

# ---------------- ENCODING ----------------
df['sex'] = df['sex'].map({"male": 0, "female": 1})
df['smoker'] = df['smoker'].map({"no": 0, "yes": 1})

df.rename(columns={
    'sex': 'is_female',
    'smoker': 'is_smoker'
}, inplace=True)

# ---------------- REGION ENCODING ----------------
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Ensure column exists (important for deployment)
if 'region_southeast' not in df.columns:
    df['region_southeast'] = 0

# ---------------- BMI CATEGORY ----------------
df['bmi_category'] = pd.cut(
    df['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'obesity']
)

df = pd.get_dummies(df, columns=['bmi_category'], drop_first=True)

# Ensure column exists
if 'bmi_category_obesity' not in df.columns:
    df['bmi_category_obesity'] = 0

# ---------------- FINAL FEATURES ----------------
features = [
    'age',
    'is_female',
    'bmi',
    'children',
    'is_smoker',
    'region_southeast',
    'bmi_category_obesity'
]

target = 'charges'

final_df = df[features + [target]].copy()

# ---------------- SCALING ----------------
scaler = StandardScaler()

scale_cols = ['age', 'bmi', 'children']
final_df[scale_cols] = scaler.fit_transform(final_df[scale_cols])

# ---------------- MODEL ----------------
X = final_df[features]
y = final_df[target]

model = LinearRegression()
model.fit(X, y)

# ---------------- SAVE ----------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")

# 💰 Insurance Cost Prediction using Machine Learning

## 📌 Overview

This project aims to predict **medical insurance costs** based on an individual’s personal and lifestyle attributes.
It is a **regression problem** where the target variable is continuous (insurance charges).

The goal is to help insurance companies and individuals estimate costs more accurately using data-driven insights.

## 🚀 Problem Statement

Insurance costs vary significantly depending on factors like age, BMI, smoking habits, and region.
This project builds a machine learning model to **predict insurance charges** and understand key factors influencing cost.

## 📊 Dataset

* Source: Kaggle (Medical Cost Personal Dataset)
* Features used:

  * Age
  * Sex
  * BMI
  * Number of Children
  * Smoker (Yes/No)
  * Region
* Target:

  * Insurance Charges

## 🔍 Exploratory Data Analysis (EDA)

* Analyzed distribution of insurance charges
* Identified strong correlation between **smoking and high charges**
* Checked relationship between BMI and cost
* Visualized trends using histograms, scatter plots, and box plots

## ⚙️ Data Preprocessing

* Handled missing values (if any)
* Encoded categorical variables using **One-Hot Encoding**
* Feature scaling applied where necessary
* Split dataset into training and testing sets

## 🤖 Model Building

The following regression models were trained and evaluated:

* Linear Regression
* Random Forest Regressor

## 💡 Key Insights

* Smoking is the **most significant factor** affecting insurance cost
* Higher BMI tends to increase charges
* Age has a positive correlation with cost

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn

## 📦 Project Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Model Training
5. Model Evaluation
6. Prediction

## 📸 Sample Output
<img width="1081" height="850" alt="image" src="https://github.com/user-attachments/assets/04d2c848-8472-4025-a32d-ed6dcb9eb4c2" />


## 🌐 Future Improvements

* Deploy model using **Streamlit**
* Use advanced models like XGBoost
* Hyperparameter tuning for better accuracy

## 📂 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/insurance-cost-predictor.git

# Navigate to the project folder
cd insurance-cost-predictor

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
```

---

## 👤 Author

Vivek Pandey

B.Tech CSE (AI)

---

## ⭐ If you found this project useful, consider giving it a star!


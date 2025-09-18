
# Employee Attrition Prediction
# Step-by-step implementation based on DA External Exam instructions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset
# df = pd.read_csv("path_to_dataset.csv")
# Example: df = pd.read_csv("data.csv")

# 2. Preprocess Dataset
# Encode categorical variables
# label_encoder = LabelEncoder()
# df['column'] = label_encoder.fit_transform(df['column'])

# 3. Handle Missing Values
# imputer = SimpleImputer(strategy='mean')  # or 'most_frequent' for categorical
# df[['numeric_column']] = imputer.fit_transform(df[['numeric_column']])

# 4. Train-Test Split
# X = df.drop('target_column', axis=1)
# y = df['target_column']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Choose Proper Attributes
# (Already handled in train-test split)

# 6. Choose Suitable Algorithm
# model = LogisticRegression()  # Or RandomForestRegressor() for regression

# 7. Train the Model
# model.fit(X_train, y_train)

# 8. Prediction on Unseen Data
# y_pred = model.predict(X_test)

# 9. Accuracy/Performance
# print("Accuracy:", accuracy_score(y_test, y_pred))  # Classification
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))  # Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('airline_passenger_satisfaction.csv')  

# Initial look
print("Initial Data Overview:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Drop unnecessary columns if present
df.drop(['id', 'Unnamed: 0'], axis=1, errors='ignore', inplace=True)

# Encode target label
df['satisfaction'] = df['satisfaction'].map({'neutral or dissatisfied': 0, 'satisfied': 1})

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[df.columns] = imputer.fit_transform(df)

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Split features and target
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------
# Logistic Regression
# ---------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, y_pred_log))
print("AUC Score:", roc_auc_score(y_test, y_prob_log))

# Confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_log)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob_log):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Feature importance (logistic regression coefficients)
coef_df = pd.Series(log_model.coef_[0], index=X.columns).sort_values()
plt.figure(figsize=(10, 6))
coef_df.plot(kind='barh')
plt.title("Feature Importance - Logistic Regression")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

# ---------------------
# Random Forest (Feature Insights)
# ---------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Top Drivers of Customer Satisfaction (Random Forest)")
plt.tight_layout()
plt.show()

print("\nTop 10 Drivers of Satisfaction (Random Forest):")
print(feature_importance_df.head(10))

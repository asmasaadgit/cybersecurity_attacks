import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score
from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================
# 🔹 STEP 1: Load Dataset
# ===========================
file_path = "cybersecurity_attacks.csv"
df = pd.read_csv(file_path)

# Sample 10,000 records for better performance
df_sampled = df.sample(n=10000, random_state=42)

# ===========================
# 🔹 STEP 2: Data Preprocessing
# ===========================
columns_to_drop = [
    "Malware Indicators", "Alerts/Warnings", "Proxy Information", 
    "Firewall Logs", "IDS/IPS Alerts", "Payload Data", 
    "User Information", "Device Information", "Timestamp", "Source IP Address", "Destination IP Address"
]
df_cleaned = df_sampled.drop(columns=columns_to_drop)

# Encode categorical variables using one-hot encoding
categorical_columns = ["Protocol", "Packet Type", "Traffic Type", "Action Taken", "Severity Level", "Log Source"]
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_columns, drop_first=True)

# Normalize numerical features using StandardScaler
numerical_columns = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
scaler = StandardScaler()
df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# Define features (X) and target variable (y)
target_column = "Attack Type"
X = df_encoded.drop(columns=[target_column, "Attack Signature", "Network Segment", "Geo-location Data"])
y = df_encoded[target_column].astype("category").cat.codes  # Convert attack types into numerical labels

# ===========================
# 🔹 STEP 3: Feature Selection (SHAP)
# ===========================
print("🔹 Running SHAP Feature Selection...")

xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=10, random_state=42, tree_method="hist", device="cpu")
xgb.fit(X, y)

explainer = shap.Explainer(xgb)
shap_values = explainer(X)

# Compute SHAP importance with restored settings
feature_importance = pd.Series(np.abs(shap_values.values).mean(axis=(0, 2)), index=X.columns)

# Remove weak features (Threshold = 0.01 to keep more features)
important_features = feature_importance[feature_importance > 0.01].index.tolist()
X = X[important_features]

print(f"✅ Selected {len(important_features)} important features.")

# ===========================
# 🔹 STEP 4: Handle Class Imbalance (SMOTE + Tomek Links)
# ===========================
print("🔹 Applying SMOTE + Tomek to balance data...")

smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

print("✅ Data balancing completed.")

# ===========================
# 🔹 STEP 5: Apply PCA
# ===========================
num_pca_components = min(10, len(important_features))
pca = PCA(n_components=num_pca_components)
X_pca = pca.fit_transform(X_resampled)

print(f"✅ PCA applied with {num_pca_components} components.")

# ===========================
# 🔹 STEP 6: Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# ===========================
# 🔹 STEP 7: Model Training & Evaluation
# ===========================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost (Tuned)": XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=15, random_state=42, tree_method="hist"),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=200, learning_rate=0.05, depth=10, verbose=0, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
    }

# ===========================
# 🔹 STEP 8: Save Results & Visualization
# ===========================
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.to_excel("optimized_model_results_12.xlsx")

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df["Accuracy"])
plt.xticks(rotation=30, ha="right")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

print("✅ Results saved to 'optimized_model_results.xlsx'")

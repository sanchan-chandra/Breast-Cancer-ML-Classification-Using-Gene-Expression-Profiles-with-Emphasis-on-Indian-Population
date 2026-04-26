import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
DATASET_PATH = "shapfeatures.csv"
df = pd.read_csv(DATASET_PATH).dropna()

# Features & labels
X = df.drop(columns=['label'])
y = LabelEncoder().fit_transform(df['label'])

feature_names = X.columns

# Train model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)

# Feature importance
importance = pd.Series(rf.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=False)

# Save results
importance.head(50).to_csv("top50_RF_biomarkers.csv")

print("Top 10 Features:")
print(importance.head(10))
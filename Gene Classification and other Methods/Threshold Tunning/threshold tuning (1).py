import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
DATASET_PATH = "shapfeatures.csv"
df = pd.read_csv(DATASET_PATH).dropna()

X = df.drop(columns=['label'])
y = LabelEncoder().fit_transform(df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train model
model = SVC(probability=True)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

# Threshold tuning
thresholds = np.arange(0.1, 0.91, 0.01)
results = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    results.append({
        'threshold': t,
        'accuracy': accuracy_score(y_test, y_pred),
        'tumor_recall': recall_score(y_test, y_pred, pos_label=1),
        'normal_recall': recall_score(y_test, y_pred, pos_label=0),
        'macro_f1': f1_score(y_test, y_pred, average='macro')
    })

df_results = pd.DataFrame(results)
df_results.to_csv("threshold_tuning.csv", index=False)

best = df_results.loc[df_results['macro_f1'].idxmax()]
print("Best Threshold:", best['threshold'])
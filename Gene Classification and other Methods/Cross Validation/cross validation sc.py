import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load dataset
DATASET_PATH = "shapfeatures.csv"
df = pd.read_csv(DATASET_PATH).dropna()

X = df.drop(columns=['label'])
y = LabelEncoder().fit_transform(df['label'])

# Model
model = SVC(kernel='rbf', probability=True)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
f1  = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')

print("Mean AUC:", auc.mean())
print("Mean Accuracy:", acc.mean())
print("Mean F1:", f1.mean())
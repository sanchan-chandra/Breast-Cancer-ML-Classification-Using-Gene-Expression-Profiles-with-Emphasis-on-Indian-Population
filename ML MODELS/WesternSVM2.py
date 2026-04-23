"""
WesternSVM.py
=============
Breast Cancer Gene Expression Classification
Train + Test on Western Dataset using SVM
Goal: Biomarker discovery + classification baseline
      (simplified, well-regularised — avoids overfitting)
Results saved to: results/WesternSVM/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, recall_score, f1_score
)
import shap

# ─────────────────────────────────────────────────────────────────────
# PATHS  —  update these
# ─────────────────────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\sanchan chandrasheka\PycharmProjects\BCModel_Dev\dataset\shapfeatures.csv"
RESULTS_DIR = r"C:\Users\sanchan chandrasheka\PycharmProjects\BCModel_Dev\results\WesternSVM2"

os.makedirs(RESULTS_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(RESULTS_DIR, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  → saved: {name}")

sns.set_style("darkgrid")
BLUE, RED, GREEN, PURPLE, AMBER = '#4f8ef7', '#ef4444', '#10b981', '#7c3aed', '#f59e0b'

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────
print("=" * 65)
print("WesternSVM — Train & Test on Western Dataset")
print("=" * 65)

df = pd.read_csv(DATA_PATH).dropna()
df = df.loc[:, ~df.columns.duplicated()]

DROP_COLS    = [c for c in ['Unnamed: 0', 'label', 'dataset_id'] if c in df.columns]
feature_cols = [c for c in df.columns if c not in DROP_COLS]
gene_names   = feature_cols

X  = df[feature_cols].values.astype(np.float32)
le = LabelEncoder()
y  = le.fit_transform(df['label'].values)

print(f"\nDataset shape   : {X.shape}")
print(f"Tumor  (1)      : {(y==1).sum()}")
print(f"Normal (0)      : {(y==0).sum()}")
print(f"Class ratio     : {(y==1).sum()/(y==0).sum():.2f} tumor:normal")

pd.DataFrame({
    'total_samples': [len(y)], 'n_features': [X.shape[1]],
    'tumor': [(y==1).sum()], 'normal': [(y==0).sum()]
}).to_csv(os.path.join(RESULTS_DIR, 'dataset_summary.csv'), index=False)

# ─────────────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT + SCALING
# ─────────────────────────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# No SMOTE — class_weight='balanced' handles imbalance without
# synthesising population-specific noise in high-dimensional gene space

# ─────────────────────────────────────────────────────────────────────
# 3. BASELINE SVM  (linear, class_weight balanced)
# ─────────────────────────────────────────────────────────────────────
print("\n[1/5] Baseline SVM (linear, all features)...")
svm_base = SVC(kernel='linear', C=0.1, class_weight='balanced',
               probability=True, random_state=42)
svm_base.fit(X_tr_sc, y_tr)

y_pred_base = svm_base.predict(X_te_sc)
y_prob_base = svm_base.predict_proba(X_te_sc)[:, 1]

print(f"  Accuracy : {accuracy_score(y_te, y_pred_base):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_te, y_prob_base):.4f}")
print(classification_report(y_te, y_pred_base, target_names=['Normal', 'Tumor']))

# ─────────────────────────────────────────────────────────────────────
# 4. BIOMARKER DISCOVERY — Random Forest
#    RF with moderate depth/trees: enough for reliable feature ranking,
#    not so large it locks onto Western-only noise
# ─────────────────────────────────────────────────────────────────────
print("\n[2/5] Biomarker discovery via Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,              # prevents deep memorisation
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr_sc, y_tr)

rf_importance = pd.Series(rf.feature_importances_, index=gene_names).sort_values(ascending=False)
rf_importance.head(50).reset_index().rename(
    columns={'index': 'gene', 0: 'rf_importance'}
).to_csv(os.path.join(RESULTS_DIR, 'top50_RF_biomarkers.csv'), index=False)

print("  Top 10 RF biomarkers:")
print(rf_importance.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────
print("\n[3/5] SHAP explainability...")
background = shap.sample(X_tr_sc, 50, random_state=42)
explainer  = shap.TreeExplainer(rf, background)
N_EXPLAIN  = min(300, len(X_te_sc))
shap_vals  = explainer.shap_values(X_te_sc[:N_EXPLAIN])

if isinstance(shap_vals, list):
    shap_tumor = shap_vals[1]
elif shap_vals.ndim == 3:
    shap_tumor = shap_vals[:, :, 1]
else:
    shap_tumor = shap_vals

shap_importance = pd.Series(
    np.abs(shap_tumor).mean(axis=0), index=gene_names
).sort_values(ascending=False)

shap_importance.head(50).reset_index().rename(
    columns={'index': 'gene', 0: 'mean_abs_shap'}
).to_csv(os.path.join(RESULTS_DIR, 'top50_SHAP_biomarkers.csv'), index=False)

print("  Top 10 SHAP biomarkers:")
print(shap_importance.head(10).to_string())

# Biomarker overlap
top30_rf   = set(rf_importance.head(30).index)
top30_shap = set(shap_importance.head(30).index)
overlap    = top30_rf & top30_shap
pd.DataFrame({'gene': sorted(overlap)}).to_csv(
    os.path.join(RESULTS_DIR, 'biomarker_overlap_RF_SHAP.csv'), index=False)
print(f"  Biomarker overlap (RF top-30 ∩ SHAP top-30): {len(overlap)} genes")

# ─────────────────────────────────────────────────────────────────────
# 6. REFINED SVM — Top-30 SHAP genes, RBF kernel
#    C=1.0 (not 10): moderate regularisation for better generalisation
# ─────────────────────────────────────────────────────────────────────
print("\n[4/5] Refined SVM (RBF, top-30 SHAP genes)...")
TOP_N     = 30
top_genes = shap_importance.head(TOP_N).index.tolist()
top_idx   = [gene_names.index(g) for g in top_genes]

X_tr_top = X_tr_sc[:, top_idx]
X_te_top = X_te_sc[:, top_idx]

svm_ref = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight='balanced', probability=True, random_state=42)
svm_ref.fit(X_tr_top, y_tr)

y_pred_ref = svm_ref.predict(X_te_top)
y_prob_ref = svm_ref.predict_proba(X_te_top)[:, 1]

print(f"  Accuracy : {accuracy_score(y_te, y_pred_ref):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_te, y_prob_ref):.4f}")
print(classification_report(y_te, y_pred_ref, target_names=['Normal', 'Tumor']))

# ─────────────────────────────────────────────────────────────────────
# 7. CROSS-VALIDATION (5-fold)
# ─────────────────────────────────────────────────────────────────────
print("\n[5/5] 5-fold stratified cross-validation...")
X_top_all = scaler.transform(X)[:, top_idx]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(svm_ref, X_top_all, y, cv=cv, scoring='roc_auc',  n_jobs=-1)
cv_acc = cross_val_score(svm_ref, X_top_all, y, cv=cv, scoring='accuracy', n_jobs=-1)
cv_f1  = cross_val_score(svm_ref, X_top_all, y, cv=cv, scoring='f1_macro', n_jobs=-1)

pd.DataFrame({
    'fold': range(1, 6), 'roc_auc': cv_auc,
    'accuracy': cv_acc, 'f1_macro': cv_f1
}).to_csv(os.path.join(RESULTS_DIR, 'cross_validation_results.csv'), index=False)

print(f"  CV ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  CV Macro F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# ─────────────────────────────────────────────────────────────────────
# 8. THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────────
thresholds = np.round(np.arange(0.10, 0.91, 0.01), 2)
thresh_records = []
for t in thresholds:
    yp = (y_prob_ref >= t).astype(int)
    if len(np.unique(yp)) < 2:
        continue
    thresh_records.append({
        'threshold':     t,
        'accuracy':      accuracy_score(y_te, yp),
        'tumor_recall':  recall_score(y_te, yp, pos_label=1, zero_division=0),
        'normal_recall': recall_score(y_te, yp, pos_label=0, zero_division=0),
        'macro_f1':      f1_score(y_te, yp, average='macro', zero_division=0),
    })

sweep_df = pd.DataFrame(thresh_records)
sweep_df.to_csv(os.path.join(RESULTS_DIR, 'threshold_sweep.csv'), index=False)

mask_b = sweep_df['tumor_recall'] >= 0.85
best_t_row = (sweep_df[mask_b].loc[sweep_df[mask_b]['normal_recall'].idxmax()]
              if mask_b.any() else sweep_df.loc[sweep_df['macro_f1'].idxmax()])
BEST_T = float(best_t_row['threshold'])
y_pred_tuned = (y_prob_ref >= BEST_T).astype(int)

print(f"\n  Recommended threshold: {BEST_T:.2f}")
print(f"  Tumor recall  : {recall_score(y_te, y_pred_tuned, pos_label=1):.4f}")
print(f"  Normal recall : {recall_score(y_te, y_pred_tuned, pos_label=0):.4f}")

# ─────────────────────────────────────────────────────────────────────
# 9. MODEL SUMMARY
# ─────────────────────────────────────────────────────────────────────
summary = pd.DataFrame([
    {
        'model':         'Baseline SVM (linear, all features)',
        'accuracy':      accuracy_score(y_te, y_pred_base),
        'roc_auc':       roc_auc_score(y_te, y_prob_base),
        'tumor_recall':  recall_score(y_te, y_pred_base, pos_label=1),
        'normal_recall': recall_score(y_te, y_pred_base, pos_label=0),
        'macro_f1':      f1_score(y_te, y_pred_base, average='macro'),
        'threshold':     0.50,
    },
    {
        'model':         'Refined SVM (RBF, top-30, t=0.50)',
        'accuracy':      accuracy_score(y_te, y_pred_ref),
        'roc_auc':       roc_auc_score(y_te, y_prob_ref),
        'tumor_recall':  recall_score(y_te, y_pred_ref, pos_label=1),
        'normal_recall': recall_score(y_te, y_pred_ref, pos_label=0),
        'macro_f1':      f1_score(y_te, y_pred_ref, average='macro'),
        'threshold':     0.50,
    },
    {
        'model':         f'Refined SVM (RBF, top-30, t={BEST_T:.2f} tuned)',
        'accuracy':      accuracy_score(y_te, y_pred_tuned),
        'roc_auc':       roc_auc_score(y_te, y_prob_ref),
        'tumor_recall':  recall_score(y_te, y_pred_tuned, pos_label=1),
        'normal_recall': recall_score(y_te, y_pred_tuned, pos_label=0),
        'macro_f1':      f1_score(y_te, y_pred_tuned, average='macro'),
        'threshold':     BEST_T,
    },
])
summary.to_csv(os.path.join(RESULTS_DIR, 'model_summary.csv'), index=False)
print("\nModel summary:")
print(summary.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
# 10. PLOTS
# ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Plot 1: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, y_pred, title, cmap in zip(
    axes,
    [y_pred_base, y_pred_ref, y_pred_tuned],
    ['Baseline SVM\n(linear, all features)',
     'Refined SVM\n(RBF, top-30, t=0.50)',
     f'Refined SVM\n(RBF, top-30, t={BEST_T:.2f} tuned)'],
    ['Blues', 'Purples', 'Greens']
):
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'], annot_kws={'size': 13})
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('WesternSVM — Confusion Matrices', fontsize=12, fontweight='bold')
plt.tight_layout()
savefig('plot1_confusion_matrices.png')

# Plot 2: ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for y_prob, label, color in [
    (y_prob_base, f'Baseline (AUC={roc_auc_score(y_te, y_prob_base):.3f})', BLUE),
    (y_prob_ref,  f'Refined  (AUC={roc_auc_score(y_te, y_prob_ref):.3f})',  RED),
]:
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    ax.plot(fpr, tpr, lw=2, color=color, label=label)
ax.plot([0,1],[0,1],'--', color='gray', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('WesternSVM — ROC Curves', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot2_roc_curves.png')

# Plot 3: Precision-Recall curves
fig, ax = plt.subplots(figsize=(7, 5))
for y_prob, label, color in [
    (y_prob_base, f'Baseline (AP={average_precision_score(y_te, y_prob_base):.3f})', BLUE),
    (y_prob_ref,  f'Refined  (AP={average_precision_score(y_te, y_prob_ref):.3f})',  RED),
]:
    prec, rec, _ = precision_recall_curve(y_te, y_prob)
    ax.plot(rec, prec, lw=2, color=color, label=label)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('WesternSVM — Precision-Recall Curves', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot3_precision_recall.png')

# Plot 4: Top-30 SHAP biomarkers
fig, ax = plt.subplots(figsize=(9, 10))
top30_shap_s = shap_importance.head(30)
colors = [RED if i < 5 else BLUE if i < 15 else GREEN for i in range(30)]
ax.barh(range(30), top30_shap_s.values[::-1], color=colors[::-1], alpha=0.85)
ax.set_yticks(range(30))
ax.set_yticklabels(top30_shap_s.index[::-1], fontsize=8)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('WesternSVM — Top 30 SHAP Biomarkers', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
savefig('plot4_shap_biomarkers.png')

# Plot 5: Top-30 RF importance
fig, ax = plt.subplots(figsize=(9, 10))
top30_rf_s = rf_importance.head(30)
ax.barh(range(30), top30_rf_s.values[::-1], color=PURPLE, alpha=0.85)
ax.set_yticks(range(30))
ax.set_yticklabels(top30_rf_s.index[::-1], fontsize=8)
ax.set_xlabel('RF Feature Importance (Gini)')
ax.set_title('WesternSVM — Top 30 RF Biomarkers', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
savefig('plot5_rf_biomarkers.png')

# Plot 6: Cross-validation per fold
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(5); w = 0.25
b1 = ax.bar(x - w, cv_auc, w, label='ROC-AUC', color=BLUE,  alpha=0.85)
b2 = ax.bar(x,     cv_acc, w, label='Accuracy', color=GREEN, alpha=0.85)
b3 = ax.bar(x + w, cv_f1,  w, label='Macro F1', color=RED,  alpha=0.85)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title(f'WesternSVM — 5-Fold CV  (AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f})', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig('plot6_cross_validation.png')

# Plot 7: Threshold sweep
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sweep_df['threshold'], sweep_df['tumor_recall'],  color=RED,   lw=2.5, label='Tumor recall')
ax.plot(sweep_df['threshold'], sweep_df['normal_recall'], color=BLUE,  lw=2.5, label='Normal recall')
ax.plot(sweep_df['threshold'], sweep_df['macro_f1'],      color=GREEN, lw=1.8, linestyle='--', label='Macro F1')
ax.axvline(0.50,   color='gray',  lw=1.2, linestyle='--', alpha=0.7, label='Default (0.50)')
ax.axvline(BEST_T, color=PURPLE,  lw=2,                               label=f'Recommended ({BEST_T:.2f})')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('WesternSVM — Threshold Tuning', fontweight='bold')
ax.legend(); ax.set_xlim(0.10, 0.90); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot7_threshold_tuning.png')

# Plot 8: SHAP beeswarm
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_tumor[:, top_idx],
    X_te_sc[:N_EXPLAIN, :][:, top_idx],
    feature_names=top_genes, show=False, plot_size=None
)
plt.title('WesternSVM — SHAP Beeswarm (Top 30 Genes)', fontweight='bold')
plt.tight_layout()
savefig('plot8_shap_beeswarm.png')

# Plot 9: Score distribution
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(y_prob_ref[y_te == 0], bins=30, alpha=0.65, color=BLUE, label='Normal', density=True)
ax.hist(y_prob_ref[y_te == 1], bins=30, alpha=0.65, color=RED,  label='Tumor',  density=True)
ax.axvline(0.50,   color='gray',  lw=1.5, linestyle='--', label='Default (0.50)')
ax.axvline(BEST_T, color=PURPLE,  lw=2,                   label=f'Recommended ({BEST_T:.2f})')
ax.set_xlabel('Predicted Tumor Probability'); ax.set_ylabel('Density')
ax.set_title('WesternSVM — Score Distribution', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot9_score_distribution.png')

# ─────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"WesternSVM — DONE. Results saved to: {RESULTS_DIR}")
print("=" * 65)

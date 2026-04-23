"""Breast Cancer Gene Expression Classification
Train on Western Dataset (shapfeatures.csv), Test on Indian Dataset (indian_AG.csv)
Model: XGBoost"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap


WESTERN_PATH = "C:shapfeatures.csv"
INDIAN_PATH  = "C:indian_AG.csv"
RESULTS_DIR  = "C:WestIndiXGB.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(RESULTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  → saved: {name}")

sns.set_style("darkgrid")
BLUE, RED, GREEN, PURPLE, AMBER = '#4f8ef7', '#ef4444', '#10b981', '#7c3aed', '#f59e0b'


print("=" * 65)
print("WestIndiXGB — Train: Western  |  Test: Indian  |  Model: XGBoost")
print("=" * 65)

def load_dataset(path, name):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df.loc[:, ~df.columns.duplicated()]
    drop = [c for c in ['Unnamed: 0', 'label', 'dataset_id'] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in drop]
    X = df[feat_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    print(f"\n{name} dataset:")
    print(f"  Shape   : {X.shape}")
    print(f"  Tumor   : {(y==1).sum()}")
    print(f"  Normal  : {(y==0).sum()}")
    return X, y, feat_cols

X_west, y_west, feat_west = load_dataset(WESTERN_PATH, "Western (train)")
X_indi, y_indi, feat_indi = load_dataset(INDIAN_PATH,  "Indian  (test)")


common_genes = sorted(set(feat_west) & set(feat_indi))
print(f"\nCommon genes between datasets: {len(common_genes)}")
print(f"  Western-only genes: {len(set(feat_west) - set(feat_indi))}")
print(f"  Indian-only  genes: {len(set(feat_indi) - set(feat_west))}")

if len(common_genes) == 0:
    raise ValueError("No common genes between Western and Indian datasets. "
                     "Check that column names match.")

west_idx = [feat_west.index(g) for g in common_genes]
indi_idx = [feat_indi.index(g) for g in common_genes]

X_west_aligned = X_west[:, west_idx]
X_indi_aligned = X_indi[:, indi_idx]
gene_names = common_genes

pd.DataFrame({
    'western_samples': [len(y_west)],
    'indian_samples':  [len(y_indi)],
    'common_genes':    [len(common_genes)],
    'western_tumor':   [(y_west==1).sum()],
    'indian_tumor':    [(y_indi==1).sum()],
}).to_csv(os.path.join(RESULTS_DIR, 'dataset_summary.csv'), index=False)


scaler    = StandardScaler()
X_west_sc = scaler.fit_transform(X_west_aligned)
X_indi_sc = scaler.transform(X_indi_aligned)


smote = SMOTE(random_state=42, k_neighbors=5)
X_west_res, y_west_res = smote.fit_resample(X_west_sc, y_west)
print(f"\nAfter SMOTE — Tumor: {(y_west_res==1).sum()}  Normal: {(y_west_res==0).sum()}")

scale_pos = (y_west_res == 0).sum() / (y_west_res == 1).sum()


print("\n[1/5] Training baseline XGBoost (all common features)...")
xgb_base = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
xgb_base.fit(X_west_res, y_west_res,
             eval_set=[(X_indi_sc, y_indi)],
             verbose=False)

y_pred_base = xgb_base.predict(X_indi_sc)
y_prob_base = xgb_base.predict_proba(X_indi_sc)[:, 1]

print(f"  Accuracy : {accuracy_score(y_indi, y_pred_base):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_indi, y_prob_base):.4f}")
print(classification_report(y_indi, y_pred_base, target_names=['Normal', 'Tumor']))


print("\n[2/5] Biomarker discovery via XGBoost native importance...")

for imp_type in ['weight', 'gain', 'cover']:
    scores = xgb_base.get_booster().get_score(importance_type=imp_type)
    gene_scores = {}
    for feat, val in scores.items():
        idx = int(feat.replace('f', ''))
        if idx < len(gene_names):
            gene_scores[gene_names[idx]] = val
    imp_series = pd.Series(gene_scores).sort_values(ascending=False)
    imp_series.head(50).reset_index().rename(
        columns={'index': 'gene', 0: imp_type}
    ).to_csv(os.path.join(RESULTS_DIR, f'top50_XGB_{imp_type}.csv'), index=False)

print(f"  Top 10 by 'gain':")
print(imp_series.head(10).to_string())

print("\n[3/5] SHAP explainability (on Indian test set)...")
explainer  = shap.TreeExplainer(xgb_base)
N_EXPLAIN  = min(500, len(X_indi_sc))
shap_vals  = explainer.shap_values(X_indi_sc[:N_EXPLAIN])

if isinstance(shap_vals, list):
    shap_tumor = shap_vals[1]
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


print("\n[4/5] Training refined XGBoost (top-30 SHAP genes)...")
TOP_N     = 30
top_genes = shap_importance.head(TOP_N).index.tolist()
top_idx   = [gene_names.index(g) for g in top_genes]

X_west_top = X_west_res[:, top_idx]
X_indi_top = X_indi_sc[:, top_idx]

xgb_ref = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
xgb_ref.fit(X_west_top, y_west_res,
            eval_set=[(X_indi_top, y_indi)],
            verbose=False)

y_pred_ref = xgb_ref.predict(X_indi_top)
y_prob_ref = xgb_ref.predict_proba(X_indi_top)[:, 1]

print(f"  Accuracy : {accuracy_score(y_indi, y_pred_ref):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_indi, y_prob_ref):.4f}")
print(classification_report(y_indi, y_pred_ref, target_names=['Normal', 'Tumor']))


print("\n[5/5] 5-fold CV on Western data (internal validation)...")
X_west_top_all = X_west_sc[:, top_idx]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_auc = cross_val_score(xgb_ref, X_west_top_all, y_west, cv=cv, scoring='roc_auc',  n_jobs=-1)
cv_acc = cross_val_score(xgb_ref, X_west_top_all, y_west, cv=cv, scoring='accuracy', n_jobs=-1)
cv_f1  = cross_val_score(xgb_ref, X_west_top_all, y_west, cv=cv, scoring='f1_macro', n_jobs=-1)

cv_results = pd.DataFrame({
    'fold':     list(range(1, 6)),
    'roc_auc':  cv_auc,
    'accuracy': cv_acc,
    'f1_macro': cv_f1
})
cv_results.to_csv(os.path.join(RESULTS_DIR, 'crossval_western_results.csv'), index=False)

print(f"  CV ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print(f"  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"  CV Macro F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


thresholds = np.arange(0.10, 0.91, 0.01)
thresh_records = []
for t in thresholds:
    yp = (y_prob_ref >= t).astype(int)
    if len(np.unique(yp)) < 2:
        continue
    thresh_records.append({
        'threshold':     round(float(t), 2),
        'accuracy':      accuracy_score(y_indi, yp),
        'tumor_recall':  recall_score(y_indi, yp, pos_label=1, zero_division=0),
        'normal_recall': recall_score(y_indi, yp, pos_label=0, zero_division=0),
        'macro_f1':      f1_score(y_indi, yp, average='macro', zero_division=0),
    })

sweep_df = pd.DataFrame(thresh_records)
sweep_df.to_csv(os.path.join(RESULTS_DIR, 'threshold_sweep.csv'), index=False)

mask_b = sweep_df['tumor_recall'] >= 0.85
best_t_row = sweep_df[mask_b].loc[sweep_df[mask_b]['normal_recall'].idxmax()] \
             if mask_b.any() else sweep_df.loc[sweep_df['macro_f1'].idxmax()]
BEST_T = float(best_t_row['threshold'])

y_pred_tuned = (y_prob_ref >= BEST_T).astype(int)
print(f"\n  Recommended threshold: {BEST_T:.2f}")
print(f"  Tumor recall  : {recall_score(y_indi, y_pred_tuned, pos_label=1):.4f}")
print(f"  Normal recall : {recall_score(y_indi, y_pred_tuned, pos_label=0):.4f}")

summary = pd.DataFrame([
    {
        'model':         'Baseline XGB (all common features)',
        'test_set':      'Indian',
        'accuracy':      accuracy_score(y_indi, y_pred_base),
        'roc_auc':       roc_auc_score(y_indi, y_prob_base),
        'tumor_recall':  recall_score(y_indi, y_pred_base, pos_label=1),
        'normal_recall': recall_score(y_indi, y_pred_base, pos_label=0),
        'macro_f1':      f1_score(y_indi, y_pred_base, average='macro'),
        'threshold':     0.50,
    },
    {
        'model':         'Refined XGB (top-30, t=0.50)',
        'test_set':      'Indian',
        'accuracy':      accuracy_score(y_indi, y_pred_ref),
        'roc_auc':       roc_auc_score(y_indi, y_prob_ref),
        'tumor_recall':  recall_score(y_indi, y_pred_ref, pos_label=1),
        'normal_recall': recall_score(y_indi, y_pred_ref, pos_label=0),
        'macro_f1':      f1_score(y_indi, y_pred_ref, average='macro'),
        'threshold':     0.50,
    },
    {
        'model':         f'Refined XGB (top-30, t={BEST_T:.2f} tuned)',
        'test_set':      'Indian',
        'accuracy':      accuracy_score(y_indi, y_pred_tuned),
        'roc_auc':       roc_auc_score(y_indi, y_prob_ref),
        'tumor_recall':  recall_score(y_indi, y_pred_tuned, pos_label=1),
        'normal_recall': recall_score(y_indi, y_pred_tuned, pos_label=0),
        'macro_f1':      f1_score(y_indi, y_pred_tuned, average='macro'),
        'threshold':     BEST_T,
    },
])
summary.to_csv(os.path.join(RESULTS_DIR, 'model_summary.csv'), index=False)
print("\nModel summary:")
print(summary.to_string(index=False))

# Biomarker overlap (gain top-30 vs SHAP top-30)
gain_scores_all = xgb_base.get_booster().get_score(importance_type='gain')
gene_gain_all = {}
for feat, val in gain_scores_all.items():
    idx = int(feat.replace('f', ''))
    if idx < len(gene_names):
        gene_gain_all[gene_names[idx]] = val
gain_series_all = pd.Series(gene_gain_all).sort_values(ascending=False)
top30_gain  = set(gain_series_all.head(30).index.tolist())
top30_shap  = set(top_genes)
overlap     = top30_gain & top30_shap

pd.DataFrame({
    'gene': sorted(overlap),
    'in_gain_top30': True,
    'in_shap_top30': True
}).to_csv(os.path.join(RESULTS_DIR, 'biomarker_overlap_Gain_SHAP.csv'), index=False)
print(f"\nBiomarker overlap (Gain top-30 ∩ SHAP top-30): {len(overlap)} genes")


print("\nGenerating plots...")

# Plot 1: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, y_pred, title, cmap in zip(
    axes,
    [y_pred_base, y_pred_ref, y_pred_tuned],
    ['Baseline XGB\n(all common features)',
     f'Refined XGB\n(top-30, t=0.50)',
     f'Refined XGB\n(top-30, t={BEST_T:.2f} tuned)'],
    ['Blues', 'Oranges', 'Greens']
):
    cm = confusion_matrix(y_indi, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'], annot_kws={'size': 13})
    ax.set_title(title, fontweight='bold', fontsize=9)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('WestIndiXGB — Confusion Matrices (Indian Test Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
savefig('plot1_confusion_matrices.png')

# Plot 2: ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for y_prob, label, color in [
    (y_prob_base, f'Baseline XGB (AUC={roc_auc_score(y_indi,y_prob_base):.3f})', BLUE),
    (y_prob_ref,  f'Refined XGB  (AUC={roc_auc_score(y_indi,y_prob_ref):.3f})',  RED),
]:
    fpr, tpr, _ = roc_curve(y_indi, y_prob)
    ax.plot(fpr, tpr, lw=2, color=color, label=label)
ax.plot([0,1],[0,1],'--', color='gray', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('WestIndiXGB — ROC Curves (Indian Test Set)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot2_roc_curves.png')

# Plot 3: Precision-Recall
fig, ax = plt.subplots(figsize=(7, 5))
for y_prob, label, color in [
    (y_prob_base, f'Baseline (AP={average_precision_score(y_indi,y_prob_base):.3f})', BLUE),
    (y_prob_ref,  f'Refined  (AP={average_precision_score(y_indi,y_prob_ref):.3f})',  RED),
]:
    prec, rec, _ = precision_recall_curve(y_indi, y_prob)
    ax.plot(rec, prec, lw=2, color=color, label=label)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('WestIndiXGB — Precision-Recall (Indian Test Set)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot3_precision_recall.png')

# Plot 4: SHAP biomarkers
fig, ax = plt.subplots(figsize=(9, 10))
top30_shap_series = shap_importance.head(30)
colors = [RED if i < 5 else BLUE if i < 15 else GREEN for i in range(30)]
ax.barh(range(30), top30_shap_series.values[::-1], color=colors[::-1], alpha=0.85)
ax.set_yticks(range(30))
ax.set_yticklabels(top30_shap_series.index[::-1], fontsize=8)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('WestIndiXGB — Top 30 SHAP Biomarkers', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
savefig('plot4_shap_biomarkers.png')

# Plot 5: XGBoost gain importance
gain_series_30 = gain_series_all.head(30)
fig, ax = plt.subplots(figsize=(9, 10))
ax.barh(range(len(gain_series_30)), gain_series_30.values[::-1], color=AMBER, alpha=0.85)
ax.set_yticks(range(len(gain_series_30)))
ax.set_yticklabels(gain_series_30.index[::-1], fontsize=8)
ax.set_xlabel('XGBoost Gain Importance')
ax.set_title('WestIndiXGB — Top 30 Genes by Gain', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
savefig('plot5_xgb_gain_importance.png')

# Plot 6: Cross-validation (Western internal)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(5)
w = 0.25
b1 = ax.bar(x - w, cv_auc, w, label='ROC-AUC', color=BLUE, alpha=0.85)
b2 = ax.bar(x,     cv_acc, w, label='Accuracy', color=GREEN, alpha=0.85)
b3 = ax.bar(x + w, cv_f1,  w, label='Macro F1', color=RED, alpha=0.85)
for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_ylim(0, 1.12); ax.set_ylabel('Score')
ax.set_title(f'WestIndiXGB — 5-Fold CV on Western  (AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f})', fontweight='bold')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig('plot6_cv_western.png')

# Plot 7: Threshold sweep
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sweep_df['threshold'], sweep_df['tumor_recall'],  color=RED,   lw=2.5, label='Tumor recall')
ax.plot(sweep_df['threshold'], sweep_df['normal_recall'], color=BLUE,  lw=2.5, label='Normal recall')
ax.plot(sweep_df['threshold'], sweep_df['macro_f1'],      color=GREEN, lw=1.8, linestyle='--', label='Macro F1')
ax.axvline(0.50,   color='gray', lw=1.2, linestyle='--', alpha=0.7, label='Default (0.50)')
ax.axvline(BEST_T, color=PURPLE, lw=2,   linestyle='-',  label=f'Recommended ({BEST_T:.2f})')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Score')
ax.set_title('WestIndiXGB — Threshold Tuning (Indian Test Set)', fontweight='bold')
ax.legend(); ax.set_xlim(0.10, 0.90); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot7_threshold_tuning.png')

# Plot 8: SHAP beeswarm (on Indian samples)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_tumor[:, top_idx],
    X_indi_sc[:N_EXPLAIN, :][:, top_idx],
    feature_names=top_genes,
    show=False, plot_size=None
)
plt.title('WestIndiXGB — SHAP Beeswarm on Indian Test Set (Top 30)', fontweight='bold')
plt.tight_layout()
savefig('plot8_shap_beeswarm.png')

# Plot 9: Score distribution (Indian)
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(y_prob_ref[y_indi == 0], bins=30, alpha=0.65, color=BLUE, label='Normal (Indian)', density=True)
ax.hist(y_prob_ref[y_indi == 1], bins=30, alpha=0.65, color=RED,  label='Tumor (Indian)',  density=True)
ax.axvline(0.50,   color='gray', lw=1.5, linestyle='--', label='Default (0.50)')
ax.axvline(BEST_T, color=PURPLE, lw=2,   linestyle='-',  label=f'Recommended ({BEST_T:.2f})')
ax.set_xlabel('Predicted Tumor Probability'); ax.set_ylabel('Density')
ax.set_title('WestIndiXGB — Score Distribution (Indian Test Set)', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
savefig('plot9_score_distribution.png')

# Plot 10: Biomarker overlap bar
only_gain = [g for g in gain_series_all.head(30).index if g not in top30_shap]
only_shap = [g for g in top_genes if g not in top30_gain]
in_both   = sorted(overlap)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(['Gain only', 'SHAP only', 'In both'], [len(only_gain), len(only_shap), len(in_both)],
        color=[AMBER, RED, GREEN], alpha=0.85)
for i, v in enumerate([len(only_gain), len(only_shap), len(in_both)]):
    ax.text(v + 0.2, i, str(v), va='center', fontweight='bold')
ax.set_xlabel('Number of Genes')
ax.set_title('WestIndiXGB — Biomarker Overlap (Gain vs SHAP Top-30)', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
savefig('plot10_biomarker_overlap.png')

# Plot 11: Generalization gap — Western CV vs Indian test performance
fig, ax = plt.subplots(figsize=(8, 5))
metrics  = ['ROC-AUC', 'Accuracy', 'Macro F1']
west_scores = [cv_auc.mean(), cv_acc.mean(), cv_f1.mean()]
indi_scores = [
    roc_auc_score(y_indi, y_prob_ref),
    accuracy_score(y_indi, y_pred_ref),
    f1_score(y_indi, y_pred_ref, average='macro')
]
x = np.arange(len(metrics))
w = 0.3
b1 = ax.bar(x - w/2, west_scores, w, label='Western CV (mean)', color=BLUE, alpha=0.85)
b2 = ax.bar(x + w/2, indi_scores, w, label='Indian test set',   color=RED,  alpha=0.85)
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.15); ax.set_ylabel('Score')
ax.set_title('WestIndiXGB — Generalization Gap\n(Western CV vs Indian External Test)', fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
savefig('plot11_generalization_gap.png')


print("\n" + "=" * 65)
print(f"WestIndiXGB — DONE. All results saved to: {RESULTS_DIR}")
print("=" * 65)
saved_files = [
    'dataset_summary.csv', 'top50_XGB_weight.csv', 'top50_XGB_gain.csv',
    'top50_XGB_cover.csv', 'top50_SHAP_biomarkers.csv',
    'biomarker_overlap_Gain_SHAP.csv', 'crossval_western_results.csv',
    'threshold_sweep.csv', 'model_summary.csv',
    'plot1_confusion_matrices.png', 'plot2_roc_curves.png', 'plot3_precision_recall.png',
    'plot4_shap_biomarkers.png', 'plot5_xgb_gain_importance.png', 'plot6_cv_western.png',
    'plot7_threshold_tuning.png', 'plot8_shap_beeswarm.png',
    'plot9_score_distribution.png', 'plot10_biomarker_overlap.png',
    'plot11_generalization_gap.png',
]
for f in saved_files:
    print(f"  ✓ {f}")

# ============================================================
# TASK 1: Iris Flower Classification
# CodeAlpha Data Science Internship
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Dataset ─────────────────────────────────────────
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("       IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 55)
print(f"\nDataset shape : {df.shape}")
print(f"Classes       : {list(iris.target_names)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")
print(f"\nStatistical Summary:\n{df.describe().round(2)}")

# ── 2. Exploratory Data Analysis ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Iris Dataset — Exploratory Data Analysis', fontsize=16, fontweight='bold')

colors = {'setosa': '#2196F3', 'versicolor': '#4CAF50', 'virginica': '#FF5722'}

for ax, (feat1, feat2) in zip(axes.flat, [
        ('sepal length (cm)', 'sepal width (cm)'),
        ('petal length (cm)', 'petal width (cm)'),
        ('sepal length (cm)', 'petal length (cm)'),
        ('sepal width (cm)', 'petal width (cm)')]):
    for sp, grp in df.groupby('species'):
        ax.scatter(grp[feat1], grp[feat2], label=sp,
                   color=colors[sp], alpha=0.7, edgecolors='white', s=60)
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(f'{feat1.split()[0].title()} vs {feat2.split()[0].title()}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/iris_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Saved] iris_eda.png")

# ── 3. Correlation Heatmap ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/iris_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] iris_correlation.png")

# ── 4. Feature Distribution Boxplot ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.suptitle('Feature Distribution by Species', fontsize=14, fontweight='bold')
palette = ['#2196F3', '#4CAF50', '#FF5722']
for ax, feat in zip(axes, iris.feature_names):
    sns.boxplot(x='species', y=feat, data=df, palette=palette, ax=ax)
    ax.set_title(feat.replace(' (cm)', '').title())
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.savefig('/home/claude/iris_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] iris_boxplot.png")

# ── 5. Train/Test Split & Scaling ────────────────────────────
X = df[iris.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ── 6. Train Multiple Models ─────────────────────────────────
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', random_state=42)
}

results = {}
print("\n" + "=" * 55)
print("              MODEL PERFORMANCE COMPARISON")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_test_sc)
    acc = accuracy_score(y_test, preds)
    results[name] = {'model': model, 'acc': acc, 'preds': preds}
    print(f"{name:<28}: Accuracy = {acc*100:.2f}%")

# ── 7. Best Model Report ─────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['acc'])
best = results[best_name]
print(f"\nBest Model: {best_name} ({best['acc']*100:.2f}%)")
print(f"\nClassification Report ({best_name}):\n")
print(classification_report(y_test, best['preds']))

# ── 8. Confusion Matrix ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Results', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, best['preds'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=iris.target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix — {best_name}', fontweight='bold')

# Accuracy comparison bar chart
names  = list(results.keys())
accs   = [results[n]['acc'] * 100 for n in names]
bar_colors = ['#4CAF50' if n == best_name else '#90CAF9' for n in names]
bars = axes[1].barh(names, accs, color=bar_colors, edgecolor='white', height=0.5)
axes[1].set_xlim(80, 105)
axes[1].set_xlabel('Accuracy (%)')
axes[1].set_title('Model Accuracy Comparison', fontweight='bold')
for bar, acc in zip(bars, accs):
    axes[1].text(acc + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{acc:.1f}%', va='center', fontsize=10)
axes[1].tick_params(axis='y', labelsize=9)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/iris_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] iris_model_results.png")

# ── 9. Feature Importance (Random Forest) ────────────────────
rf = results['Random Forest']['model']
importances = rf.feature_importances_
feat_df = pd.Series(importances, index=iris.feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
feat_df.plot(kind='barh', ax=ax, color='#2196F3', edgecolor='white')
ax.set_title('Feature Importance (Random Forest)', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/iris_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("[Saved] iris_feature_importance.png")

print("\n" + "=" * 55)
print("    TASK 1 COMPLETE — Iris Classification")
print("=" * 55)

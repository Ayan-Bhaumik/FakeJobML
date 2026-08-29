#!/usr/bin/env python3
"""
Complete pipeline for Fake Job Posting Detection
Runs all steps: preprocessing, feature engineering, model training, evaluation, feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
import json

warnings.filterwarnings("ignore")

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
import xgboost as xgb
from scipy.sparse import hstack

# Visualization settings
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

# Create results directory
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("FAKE JOB POSTING DETECTION - COMPLETE PIPELINE")
print("=" * 60)

# ============================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================
print("\n[1/10] Loading dataset...")
df = pd.read_csv("dataset/fake_job_postings.csv")
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\nClass distribution:")
class_dist = df["fraudulent"].value_counts()
print(class_dist)
print(f"Fraudulent %: {df['fraudulent'].mean()*100:.2f}%")

print(f"\nMissing values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing": missing, "Percentage": missing_pct})
print(missing_df[missing_df["Missing"] > 0].to_string())

# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================
print("\n[2/10] Text preprocessing...")

# Precompile regex patterns
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'http\S+|www\.\S+')
SPECIAL_CHAR_RE = re.compile(r'[^a-zA-Z\s]')
WHITESPACE_RE = re.compile(r'\s+')

def clean_text(text):
    """Clean a single text field."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = HTML_TAG_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = SPECIAL_CHAR_RE.sub(' ', text)
    text = WHITESPACE_RE.sub(' ', text).strip()
    return text

def clean_and_remove_stopwords(text):
    """Clean text and remove stopwords."""
    cleaned = clean_text(text)
    words = cleaned.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)

# Apply to all text columns
text_cols = ['title', 'description', 'requirements', 'company_profile']
for col in text_cols:
    df[f'{col}_clean'] = df[col].apply(clean_text)
    df[f'{col}_clean_nostop'] = df[col].apply(clean_and_remove_stopwords)

print("Text cleaning complete!")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[3/10] Feature engineering...")

# Combined text for TF-IDF
df['combined_text'] = (
    df['title_clean_nostop'] + ' ' +
    df['company_profile_clean_nostop'] + ' ' +
    df['description_clean_nostop'] + ' ' +
    df['requirements_clean_nostop']
)

# Text length features
df['title_length'] = df['title_clean'].str.len()
df['desc_length'] = df['description_clean'].str.len()
df['req_length'] = df['requirements_clean'].str.len()
df['profile_length'] = df['company_profile_clean'].str.len()
df['total_text_length'] = df['title_length'] + df['desc_length'] + df['req_length'] + df['profile_length']

# Categorical features
cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'department']
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# Extract country from location
df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
cat_cols.append('country')

# Label encode
label_encoders = {}
df_encoded = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Binary features
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']
df['has_salary_range'] = df['salary_range'].notna().astype(int)

# Also add to df_encoded
df_encoded['has_salary_range'] = df['has_salary_range']

print("Feature engineering complete!")

# ============================================================
# 4. TF-IDF VECTORIZATION
# ============================================================
print("\n[4/10] TF-IDF vectorization...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

tfidf_features = tfidf.fit_transform(df['combined_text'])
print(f"TF-IDF shape: {tfidf_features.shape}")
tfidf_feature_names = tfidf.get_feature_names_out()

# ============================================================
# 5. COMBINE FEATURES
# ============================================================
print("\n[5/10] Combining features...")
meta_feature_cols = [
    'employment_type_encoded', 'required_experience_encoded',
    'required_education_encoded', 'industry_encoded', 'function_encoded',
    'department_encoded', 'country_encoded',
    'telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range',
    'title_length', 'desc_length', 'req_length', 'profile_length', 'total_text_length'
]

meta_features = df_encoded[meta_feature_cols].values
scaler = StandardScaler()
meta_features_scaled = scaler.fit_transform(meta_features)

X = hstack([tfidf_features, meta_features_scaled])
y = df['fraudulent'].values

print(f"Combined feature matrix shape: {X.shape}")

# ============================================================
# 6. TRAIN/TEST SPLIT + SMOTE
# ============================================================
print("\n[6/10] Train/test split and SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class dist: {np.bincount(y_train)}")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train_resampled.shape}")
print(f"After SMOTE class dist: {np.bincount(y_train_resampled)}")

# ============================================================
# 7. DEFINE MODELS
# ============================================================
print("\n[7/10] Defining models...")

# For MultinomialNB, use only TF-IDF features (non-negative)
# For others, use combined features
models = {
    'Naive Bayes (TF-IDF only)': MultinomialNB(alpha=0.1),
    'SVM': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', n_jobs=1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', alpha=0.001, batch_size=256,
        learning_rate='adaptive', max_iter=100,
        random_state=42, early_stopping=True, validation_fraction=0.1
    )
}

# Use TF-IDF only for Naive Bayes
X_train_tfidf = X_train[:, :5000]
X_test_tfidf = X_test[:, :5000]

# Apply SMOTE to TF-IDF features as well
smote_tfidf = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled_tfidf, y_train_resampled_tfidf = smote_tfidf.fit_resample(X_train_tfidf, y_train)
print(f"After SMOTE (TF-IDF): {X_train_resampled_tfidf.shape}")

# ============================================================
# 8. TRAIN AND EVALUATE
# ============================================================
print("\n[8/10] Training and evaluating models...")

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, cv=5, use_tfidf_only=False):
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    # Select appropriate training data
    if use_tfidf_only:
        X_train_cv = X_train_resampled_tfidf
        y_train_cv = y_train_resampled_tfidf
        X_train_fit = X_train_resampled_tfidf
        y_train_fit = y_train_resampled_tfidf
        X_test_eval = X_test_tfidf
    else:
        X_train_cv = X_train
        y_train_cv = y_train
        X_train_fit = X_train
        y_train_fit = y_train
        X_test_eval = X_test

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_cv, y_train_cv, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=1)
    print(f"CV F1-scores: {cv_scores}")
    print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Train on full data
    model.fit(X_train_fit, y_train_fit)
    # Predict
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    }

    print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")
    print(f"Test F1-Score:  {metrics['f1']:.4f}")
    if metrics['auc_roc']:
        print(f"Test AUC-ROC:   {metrics['auc_roc']:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fraudulent'], zero_division=0))

    return metrics, y_pred, y_pred_proba, model

results = {}
predictions = {}
probabilities = {}
trained_models = {}

for name, model in models.items():
    use_tfidf = 'Naive Bayes' in name
    metrics, y_pred, y_pred_proba, trained_model = train_and_evaluate(
        model, X_train_resampled, y_train_resampled, X_test, y_test, name, use_tfidf_only=use_tfidf
    )
    results[name] = metrics
    predictions[name] = y_pred
    probabilities[name] = y_pred_proba
    trained_models[name] = trained_model

# ============================================================
# 9. RESULTS COMPARISON
# ============================================================
print("\n[9/10] Generating results comparison...")

results_df = pd.DataFrame(results).T
results_df = results_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']]
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
results_df = results_df.round(4)

print("\nResults Comparison Table:")
print(results_df.to_string())

results_df.to_csv('results/model_comparison.csv')
print("\nResults saved to results/model_comparison.csv")

# Bar chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
colors = sns.color_palette('husl', len(results_df))

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(results_df.index, results_df[metric], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(metric, fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('results/model_comparison_barchart.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions.items()):
    if idx >= 5:
        break
    cm = confusion_matrix(y_test, y_pred)
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fraudulent'],
                yticklabels=['Real', 'Fraudulent'],
                cbar=False)
    ax.set_title(name, fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

if len(predictions) < 6:
    axes[5].axis('off')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC Curves
plt.figure(figsize=(10, 8))
for name, y_pred_proba in probabilities.items():
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================
print("\n[10/10] Feature importance analysis...")

# Best model by F1
best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
print(f"Best model (by F1): {best_model_name}")
best_model = trained_models[best_model_name]

# Feature names
tfidf_names = [f'tfidf_{name}' for name in tfidf_feature_names]
meta_names = meta_feature_cols
all_feature_names = list(tfidf_names) + list(meta_names)

# Extract importances
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("\nTop 25 most important features:")
    print(importance_df.head(25).to_string(index=False))

    importance_df.to_csv('results/feature_importance.csv', index=False)
    print("\nFull feature importance saved to results/feature_importance.csv")

    # Visualize top 20
    top_n = 20
    top_features = importance_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
    bars = plt.barh(range(top_n), top_features['importance'], color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(range(top_n), top_features['feature'], fontsize=11)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Predictive Features for Fraud Detection\n({best_model_name})',
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        plt.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Category analysis
    def categorize_feature(name):
        if name.startswith('tfidf_'):
            return 'TF-IDF (Text)'
        elif 'length' in name.lower():
            return 'Text Length'
        elif name in ['telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range']:
            return 'Binary Meta-Feature'
        elif '_encoded' in name:
            return 'Categorical (Encoded)'
        else:
            return 'Other'

    importance_df['category'] = importance_df['feature'].apply(categorize_feature)
    cat_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
    print("\nTotal importance by category:")
    print(cat_importance)

    plt.figure(figsize=(10, 6))
    cat_importance.plot(kind='bar', color=sns.color_palette('husl', len(cat_importance)), edgecolor='black')
    plt.title('Total Feature Importance by Category', fontsize=14, fontweight='bold')
    plt.ylabel('Sum of Importances', fontsize=12)
    plt.xlabel('Feature Category', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/feature_category_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# EXPORT README TABLE
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS FOR README")
print("="*60)
print()
print("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |")
print("|---|---|---|---|---|---|")
for name in results_df.index:
    row = results_df.loc[name]
    print(f"| {name} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['AUC-ROC']:.4f} |")

with open('results/readme_table.txt', 'w') as f:
    f.write("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\n")
    f.write("|---|---|---|---|---|---|\n")
    for name in results_df.index:
        row = results_df.loc[name]
        f.write(f"| {name} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['AUC-ROC']:.4f} |\n")

print("\nTable saved to results/readme_table.txt")
print("\nAll done! Results saved in 'results/' folder.")
print("Files created:")
for f in sorted(os.listdir('results')):
    print(f"  results/{f}")
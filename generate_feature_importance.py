#!/usr/bin/env python3
"""
Generate feature importance plots using Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re

warnings.filterwarnings("ignore")
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), 'mpl_config')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack

os.makedirs("results", exist_ok=True)

print("Loading data and rebuilding features...")

# Load dataset
df = pd.read_csv("dataset/fake_job_postings.csv")

# Text preprocessing
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'http\S+|www\.\S+')
SPECIAL_CHAR_RE = re.compile(r'[^a-zA-Z\s]')
WHITESPACE_RE = re.compile(r'\s+')

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = HTML_TAG_RE.sub(' ', text)
    text = URL_RE.sub(' ', text)
    text = SPECIAL_CHAR_RE.sub(' ', text)
    text = WHITESPACE_RE.sub(' ', text).strip()
    return text

def clean_and_remove_stopwords(text):
    cleaned = clean_text(text)
    words = cleaned.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)

text_cols = ['title', 'description', 'requirements', 'company_profile']
for col in text_cols:
    df[f'{col}_clean'] = df[col].apply(clean_text)
    df[f'{col}_clean_nostop'] = df[col].apply(clean_and_remove_stopwords)

df['combined_text'] = (
    df['title_clean_nostop'] + ' ' +
    df['company_profile_clean_nostop'] + ' ' +
    df['description_clean_nostop'] + ' ' +
    df['requirements_clean_nostop']
)

df['title_length'] = df['title_clean'].str.len()
df['desc_length'] = df['description_clean'].str.len()
df['req_length'] = df['requirements_clean'].str.len()
df['profile_length'] = df['company_profile_clean'].str.len()
df['total_text_length'] = df['title_length'] + df['desc_length'] + df['req_length'] + df['profile_length']

cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'department']
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
cat_cols.append('country')

label_encoders = {}
df_encoded = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df['has_salary_range'] = df['salary_range'].notna().astype(int)
df_encoded['has_salary_range'] = df['has_salary_range']

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
tfidf_features = tfidf.fit_transform(df['combined_text'])
tfidf_feature_names = tfidf.get_feature_names_out()

# Meta features
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

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train Random Forest for feature importance
print("Training Random Forest for feature importance...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=5,
    min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=1
)
rf.fit(X_train_resampled, y_train_resampled)

# Get feature importances
importances = rf.feature_importances_
tfidf_names = [f'tfidf_{name}' for name in tfidf_feature_names]
meta_names = meta_feature_cols
all_feature_names = list(tfidf_names) + list(meta_names)

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
plt.title(f'Top {top_n} Most Predictive Features for Fraud Detection\n(Random Forest)',
          fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    plt.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('results/feature_importance_top20.png', dpi=300, bbox_inches='tight')
plt.close()
print("Top 20 feature importance plot saved!")

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
print("Category importance plot saved!")

print("\nAll feature importance analysis complete!")
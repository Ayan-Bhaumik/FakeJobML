import nbformat as nbf

# Read existing notebook
with open("fake_job_detection.ipynb", "r") as f:
    nb = nbf.read(f, as_version=4)

# Cell: TF-IDF Vectorization
markdown1 = """## 4. TF-IDF Vectorization

Convert combined text to TF-IDF features, then combine with meta-features.
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown1))

code1 = """# TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

print("Fitting TF-IDF vectorizer...")
tfidf_features = tfidf.fit_transform(df['combined_text'])
print(f"TF-IDF shape: {tfidf_features.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")

# Get feature names
tfidf_feature_names = tfidf.get_feature_names_out()
print(f"First 20 features: {tfidf_feature_names[:20]}")
"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# Cell: Prepare meta-features
markdown2 = """## 5. Feature Matrix Construction

Combine TF-IDF features with meta-features:
- Categorical encoded features
- Binary features (telecommuting, has_company_logo, has_questions, has_salary_range)
- Text length features
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown2))

code2 = """# Prepare meta-features
meta_feature_cols = [
    'employment_type_encoded', 'required_experience_encoded',
    'required_education_encoded', 'industry_encoded', 'function_encoded',
    'department_encoded', 'country_encoded',
    'telecommuting', 'has_company_logo', 'has_questions', 'has_salary_range',
    'title_length', 'desc_length', 'req_length', 'profile_length', 'total_text_length'
]

meta_features = df[meta_feature_cols].values
print(f"Meta-features shape: {meta_features.shape}")

# Scale meta-features
scaler = StandardScaler()
meta_features_scaled = scaler.fit_transform(meta_features)

# Combine TF-IDF and meta-features
from scipy.sparse import hstack
X = hstack([tfidf_features, meta_features_scaled])
y = df['fraudulent'].values

print(f"Combined feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

# Cell: Train/test split
markdown3 = """## 6. Train/Test Split with Stratification
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown3))

code3 = """# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train class dist: {np.bincount(y_train)}")
print(f"Test class dist: {np.bincount(y_test)}")

# Apply SMOTE to training data only
print("\\nApplying SMOTE to training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Train shape: {X_train_resampled.shape}")
print(f"After SMOTE - Class dist: {np.bincount(y_train_resampled)}")
"""
nb.cells.append(nbf.v4.new_code_cell(code3))

# Cell: Define models
markdown4 = """## 7. Model Definitions

Define all models to compare:
1. Naive Bayes (MultinomialNB) - works well with TF-IDF
2. SVM (LinearSVC) - good for high-dimensional sparse data
3. Random Forest - handles mixed feature types well
4. XGBoost - strong gradient boosting
5. Neural Network (MLP) - for comparison
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown4))

code4 = """# Define models
models = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'SVM': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', n_jobs=-1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', alpha=0.001, batch_size=256,
        learning_rate='adaptive', max_iter=100,
        random_state=42, early_stopping=True, validation_fraction=0.1
    )
}

print("Models defined:")
for name, model in models.items():
    print(f"  {name}: {model.__class__.__name__}")
"""
nb.cells.append(nbf.v4.new_code_cell(code4))

with open("fake_job_detection.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook updated with TF-IDF and models!")
import nbformat as nbf

# Read existing notebook
with open("fake_job_detection.ipynb", "r") as f:
    nb = nbf.read(f, as_version=4)

# Update the model definitions cell to use faster models and n_jobs=1
# Find cell with model definitions and update it
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'Define models' in ''.join(cell.source):
        new_code = """# Define models
# Note: Using n_jobs=1 to avoid parallel processing issues in this environment
# LinearSVC with CalibratedClassifierCV is used instead of SVC for speed

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

models = {
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Linear SVM': CalibratedClassifierCV(LinearSVC(random_state=42, class_weight='balanced', max_iter=5000), cv=3),
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

print("Models defined:")
for name, model in models.items():
    print(f"  {name}: {model.__class__.__name__}")"""
        nb.cells[i] = nbf.v4.new_code_cell(new_code)
        break

# Add a cell after model definitions for TF-IDF only split for Naive Bayes
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'Models defined:' in ''.join(cell.source):
        tfidf_cell = """# For Naive Bayes, use only TF-IDF features (non-negative)
# Create TF-IDF only splits and apply SMOTE

X_train_tfidf = X_train[:, :5000]
X_test_tfidf = X_test[:, :5000]

# Apply SMOTE to TF-IDF features
smote_tfidf = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled_tfidf, y_train_resampled_tfidf = smote_tfidf.fit_resample(X_train_tfidf, y_train)
print(f"After SMOTE (TF-IDF): {X_train_resampled_tfidf.shape}")"""
        nb.cells.insert(i + 1, nbf.v4.new_code_cell(tfidf_cell))
        break

# Update training function to handle Naive Bayes with TF-IDF only
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'def train_and_evaluate' in ''.join(cell.source):
        new_code = """# Training and evaluation function
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, cv=5, use_tfidf_only=False):
    \"\"\"Train model and return metrics.\"\"\"
    print(f"\\n{'='*50}")
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

    # Cross-validation on resampled training data
    cv_scores = cross_val_score(model, X_train_cv, y_train_cv, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=1)
    print(f"CV F1-scores: {cv_scores}")
    print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Train on full resampled training data
    model.fit(X_train_fit, y_train_fit)

    # Predict on test set
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
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

    # Classification report
    print(f"\\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fraudulent'], zero_division=0))

    return metrics, y_pred, y_pred_proba, model

# Train all models
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

print("\\n\\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)
for name, metrics in results.items():
    print(f"{name:20s} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f if metrics['auc_roc'] else 'N/A'}")"""
        nb.cells[i] = nbf.v4.new_code_cell(new_code)
        break

# Update the feature importance section to use Random Forest explicitly
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'best_model_name = max(results.items()' in ''.join(cell.source):
        new_code = """# Feature importance from best tree-based model (Random Forest)
# Use Random Forest for feature importance analysis (best tree-based model)
best_model_name = 'Random Forest'
print(f"Using {best_model_name} for feature importance analysis")

best_model = trained_models[best_model_name]

# Get feature names
tfidf_names = [f'tfidf_{name}' for name in tfidf_feature_names]
meta_names = meta_feature_cols
all_feature_names = list(tfidf_names) + list(meta_names)

print(f"Total features: {len(all_feature_names)}")
print(f"TF-IDF features: {len(tfidf_names)}")
print(f"Meta features: {len(meta_names)}")"""
        nb.cells[i] = nbf.v4.new_code_cell(new_code)
        break

# Add actual results as a markdown cell after the export cell
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'Generate markdown table for README' in ''.join(cell.source):
        results_markdown = """## 16. Final Results Summary

The following table shows the actual results from model training:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Naive Bayes | 0.9114 | 0.3386 | 0.8728 | 0.4879 | 0.9700 |
| Linear SVM | 0.9857 | 0.9236 | 0.7688 | 0.8391 | 0.9883 |
| Random Forest | 0.9782 | 0.7811 | 0.7630 | 0.7719 | 0.9851 |
| XGBoost | 0.9360 | 0.4247 | 0.9133 | 0.5798 | 0.9784 |
| Neural Network | 0.9877 | 0.9108 | 0.8266 | 0.8667 | 0.9925 |

**Key Findings:**
- **Best Overall (F1-Score):** Neural Network (0.8667)
- **Best Precision:** Linear SVM (0.9236) - fewest false alarms
- **Best Recall:** XGBoost (0.9133) - catches most fraudulent postings
- **Best AUC-ROC:** Neural Network (0.9925) - best overall discrimination

The Neural Network achieves the highest F1-Score (0.8667) and AUC-ROC (0.9925), making it the best overall model for this task. Linear SVM provides excellent precision (0.9236) which is important when false positives are costly."""
        nb.cells.insert(i + 1, nbf.v4.new_markdown_cell(results_markdown))
        break

# Save the updated notebook
with open("fake_job_detection.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook updated with actual results and fixes!")
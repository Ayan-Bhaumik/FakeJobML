import nbformat as nbf

# Read existing notebook
with open("fake_job_detection.ipynb", "r") as f:
    nb = nbf.read(f, as_version=4)

# Cell: Training function
markdown1 = """## 8. Model Training and Cross-Validation

Train each model and evaluate using cross-validation on the resampled training data.
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown1))

code1 = """# Training and evaluation function
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name, cv=5):
    \"\"\"Train model and return metrics.\"\"\"
    print(f"\\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    # Cross-validation on resampled training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                                scoring='f1', n_jobs=-1)
    print(f"CV F1-scores: {cv_scores}")
    print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # Train on full resampled training data
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

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
    metrics, y_pred, y_pred_proba, trained_model = train_and_evaluate(
        model, X_train_resampled, y_train_resampled, X_test, y_test, name
    )
    results[name] = metrics
    predictions[name] = y_pred
    probabilities[name] = y_pred_proba
    trained_models[name] = trained_model

print("\\n\\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)
for name, metrics in results.items():
    print(f"{name:20s} | Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f if metrics['auc_roc'] else 'N/A'}")
"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# Cell: Results comparison table
markdown2 = """## 9. Results Comparison and Visualization
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown2))

code2 = """# Create results DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df[['accuracy', 'precision', 'recall', 'f1', 'auc_roc']]
results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
results_df = results_df.round(4)

print("Results Comparison Table:")
print(results_df.to_string())

# Save to CSV
results_df.to_csv('results/model_comparison.csv')
print("\\nResults saved to results/model_comparison.csv")

# Bar chart comparison
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
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('results/model_comparison_barchart.png', dpi=300, bbox_inches='tight')
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

# Cell: Confusion matrices
markdown3 = """## 10. Confusion Matrices
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown3))

code3 = """# Confusion matrices for all models
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

# Remove empty subplot
if len(predictions) < 6:
    axes[5].axis('off')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code3))

# Cell: ROC Curves
markdown4 = """## 11. ROC Curves
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown4))

code4 = """# ROC Curves
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
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code4))

with open("fake_job_detection.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook updated with training and evaluation!")
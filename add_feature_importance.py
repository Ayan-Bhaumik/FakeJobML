import nbformat as nbf

# Read existing notebook
with open("fake_job_detection.ipynb", "r") as f:
    nb = nbf.read(f, as_version=4)

# Cell: Feature Importance Analysis
markdown1 = """## 12. Feature Importance Analysis

Extract and visualize the top features most predictive of fraudulent postings using the best tree-based model (Random Forest or XGBoost).
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown1))

code1 = """# Feature importance from best tree-based model
# Determine best model based on F1-score
best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
print(f"Best model (by F1): {best_model_name}")

best_model = trained_models[best_model_name]

# Get feature names
tfidf_names = [f'tfidf_{name}' for name in tfidf_feature_names]
meta_names = meta_feature_cols
all_feature_names = list(tfidf_names) + list(meta_names)

print(f"Total features: {len(all_feature_names)}")
print(f"TF-IDF features: {len(tfidf_names)}")
print(f"Meta features: {len(meta_names)}")
"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# Cell: Extract feature importances
code2 = """# Extract feature importances
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("Top 25 most important features:")
    print(importance_df.head(25).to_string(index=False))

    # Save full importance list
    importance_df.to_csv('results/feature_importance.csv', index=False)
    print("\\nFull feature importance saved to results/feature_importance.csv")
else:
    print("Model doesn't have feature_importances_ or coef_ attribute")
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

# Cell: Visualize top features
markdown2 = """## 13. Feature Importance Visualization
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown2))

code3 = """# Visualize top 20 features
top_n = 20
top_features = importance_df.head(top_n).iloc[::-1]  # Reverse for horizontal bar chart

plt.figure(figsize=(12, 10))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
bars = plt.barh(range(top_n), top_features['importance'], color=colors, edgecolor='black', linewidth=0.5)
plt.yticks(range(top_n), top_features['feature'], fontsize=11)
plt.xlabel('Importance', fontsize=12)
plt.title(f'Top {top_n} Most Predictive Features for Fraud Detection\\n({best_model_name})',
          fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
    plt.text(val + 0.0001, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/feature_importance_top20.png', dpi=300, bbox_inches='tight')
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code3))

# Cell: Categorize features
markdown3 = """## 14. Feature Category Analysis

Group features by type to understand what categories are most predictive.
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown3))

code4 = """# Categorize features by type
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

# Aggregate by category
cat_importance = importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
print("Total importance by category:")
print(cat_importance)

# Top features by category
print("\\nTop 5 features per category:")
for cat in importance_df['category'].unique():
    cat_features = importance_df[importance_df['category'] == cat].head(5)
    print(f"\\n{cat}:")
    for _, row in cat_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")

# Category importance plot
plt.figure(figsize=(10, 6))
cat_importance.plot(kind='bar', color=sns.color_palette('husl', len(cat_importance)), edgecolor='black')
plt.title('Total Feature Importance by Category', fontsize=14, fontweight='bold')
plt.ylabel('Sum of Importances', fontsize=12)
plt.xlabel('Feature Category', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/feature_category_importance.png', dpi=300, bbox_inches='tight')
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code4))

# Cell: Export final results for README
markdown4 = """## 15. Export Results for Paper

Generate the final comparison table to fill into README.md
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown4))

code5 = """# Generate markdown table for README
print("Markdown table for README:")
print()
print("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |")
print("|---|---|---|---|---|---|")
for name in results_df.index:
    row = results_df.loc[name]
    print(f"| {name} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['AUC-ROC']:.4f} |")

# Also save as formatted text file
with open('results/readme_table.txt', 'w') as f:
    f.write("| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |\\n")
    f.write("|---|---|---|---|---|---|\\n")
    for name in results_df.index:
        row = results_df.loc[name]
        f.write(f"| {name} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-Score']:.4f} | {row['AUC-ROC']:.4f} |\\n")

print("\\nTable saved to results/readme_table.txt")
"""
nb.cells.append(nbf.v4.new_code_cell(code5))

with open("fake_job_detection.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook updated with feature importance and export!")
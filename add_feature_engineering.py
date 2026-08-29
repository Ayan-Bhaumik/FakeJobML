import nbformat as nbf

# Read existing notebook
with open("fake_job_detection.ipynb", "r") as f:
    nb = nbf.read(f, as_version=4)

# Cell: Apply preprocessing to all text columns
code1 = """# Apply text cleaning to all text columns
text_cols = ['title', 'description', 'requirements', 'company_profile']

print("Cleaning text columns...")
for col in text_cols:
    df[f'{col}_clean'] = df[col].apply(clean_text)
    df[f'{col}_clean_nostop'] = df[col].apply(clean_and_remove_stopwords)

print("Cleaning complete!")
print(f"\\nSample of cleaned title:")
print(df['title_clean'].iloc[0])
print(f"\\nSample of cleaned description (no stopwords):")
print(df['description_clean_nostop'].iloc[0][:200])
"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# Cell: Create combined text field for TF-IDF
code2 = """# Create combined text field for TF-IDF
df['combined_text'] = (
    df['title_clean_nostop'] + ' ' +
    df['company_profile_clean_nostop'] + ' ' +
    df['description_clean_nostop'] + ' ' +
    df['requirements_clean_nostop']
)

# Text length features (meta-features)
df['title_length'] = df['title_clean'].str.len()
df['desc_length'] = df['description_clean'].str.len()
df['req_length'] = df['requirements_clean'].str.len()
df['profile_length'] = df['company_profile_clean'].str.len()
df['total_text_length'] = df['title_length'] + df['desc_length'] + df['req_length'] + df['profile_length']

print("Combined text created.")
print(f"Combined text sample (first 200 chars):")
print(df['combined_text'].iloc[0][:200])
print(f"\\nText length stats:")
print(df[['title_length', 'desc_length', 'req_length', 'profile_length', 'total_text_length']].describe())
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

# Cell: Handle categorical features
markdown3 = """## 3. Categorical Feature Encoding

Encode categorical columns:
- `employment_type`
- `required_experience`
- `required_education`
- `industry`
- `function`
- `location` (extract country)
- `department`
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown3))

code3 = """# Handle categorical features
cat_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'department']

# Fill missing with 'Unknown'
for col in cat_cols:
    df[col] = df[col].fillna('Unknown')

# Extract country from location
df['country'] = df['location'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) else 'Unknown')
cat_cols.append('country')

print("Unique values per categorical column:")
for col in cat_cols:
    print(f"  {col}: {df[col].nunique()} unique")

# Label encode categorical columns
label_encoders = {}
df_encoded = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_encoded[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("\\nEncoding complete!")
print("Encoded columns:", [f'{c}_encoded' for c in cat_cols])
"""
nb.cells.append(nbf.v4.new_code_cell(code3))

# Cell: Binary features already encoded
code4 = """# Binary features are already 0/1
binary_cols = ['telecommuting', 'has_company_logo', 'has_questions']

print("Binary feature distributions:")
for col in binary_cols:
    print(f"  {col}: {df[col].value_counts().to_dict()}")

# Salary range - create binary indicator for missing salary
df['has_salary_range'] = df['salary_range'].notna().astype(int)
print(f"\\n  has_salary_range: {df['has_salary_range'].value_counts().to_dict()}")
"""
nb.cells.append(nbf.v4.new_code_cell(code4))

with open("fake_job_detection.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook updated with feature engineering!")
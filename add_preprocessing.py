import nbformat as nbf

# Read existing notebook
with open("fake_job_detection.ipynb", "r") as f:
    nb = nbf.read(f, as_version=4)

# Cell 4: Text Preprocessing Functions
markdown_content = """## 2. Text Preprocessing

We'll clean the text fields:
- `title`, `description`, `requirements`, `company_profile`

Cleaning steps:
1. Convert to lowercase
2. Remove HTML tags
3. Remove special characters and URLs
4. Remove extra whitespace
5. Remove stopwords (using sklearn's built-in list)
"""
nb.cells.append(nbf.v4.new_markdown_cell(markdown_content))

code_content = """# Text preprocessing functions
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Precompile regex patterns
HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'http\\S+|www\\.\\S+')
SPECIAL_CHAR_RE = re.compile(r'[^a-zA-Z\\s]')
WHITESPACE_RE = re.compile(r'\\s+')

def clean_text(text):
    \"\"\"Clean a single text field.\"\"\"
    if pd.isna(text):
        return ""
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = HTML_TAG_RE.sub(' ', text)
    # Remove URLs
    text = URL_RE.sub(' ', text)
    # Remove special characters (keep only letters and spaces)
    text = SPECIAL_CHAR_RE.sub(' ', text)
    # Remove extra whitespace
    text = WHITESPACE_RE.sub(' ', text).strip()
    return text

def clean_and_remove_stopwords(text):
    \"\"\"Clean text and remove stopwords.\"\"\"
    cleaned = clean_text(text)
    words = cleaned.split()
    # Remove stopwords and very short words
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return ' '.join(words)

# Test on a sample
sample = df['description'].iloc[1]
print("Original (first 300 chars):")
print(sample[:300])
print("\\nCleaned (first 300 chars):")
cleaned_sample = clean_text(sample)
print(cleaned_sample[:300])
print("\\nCleaned + no stopwords (first 300 chars):")
no_stop = clean_and_remove_stopwords(sample)
print(no_stop[:300])
"""
nb.cells.append(nbf.v4.new_code_cell(code_content))

with open("fake_job_detection.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook updated with preprocessing!")
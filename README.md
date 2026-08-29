# A Hybrid Machine Learning Framework for Detecting Fraudulent Online Job Postings

A machine learning project that detects fraudulent job postings using text and meta-feature analysis, developed as part of a research paper submission.

---

## 📋 Quick Start for This Device

### Prerequisites (Already Installed)
- Python 3.14+ (via Homebrew)
- Virtual environment: `.venv` (in project root)

### Activate Environment
```bash
source /Users/onyx/FakeJobML/.venv/bin/activate
```

### Run Options

**Option 1: Jupyter Notebook (Interactive)**
```bash
jupyter notebook fake_job_detection.ipynb
# Then: Kernel → Restart & Run All
```

**Option 2: Fast Pipeline (Command Line, ~2-3 min)**
```bash
python run_pipeline_fast.py
```

**Option 3: Feature Importance Only**
```bash
python generate_feature_importance.py
```

### Expected Outputs (in `results/`)
| File | Description |
|------|-------------|
| `model_comparison.csv` | Metrics table (CSV) |
| `model_comparison_barchart.png` | Bar chart comparing all 5 models |
| `confusion_matrices.png` | Confusion matrices for all models |
| `roc_curves.png` | ROC curves comparison |
| `feature_importance.csv` | Full feature importance (3016 features) |
| `feature_importance_top20.png` | Top 20 features horizontal bar chart |
| `feature_category_importance.png` | Feature category importance breakdown |
| `readme_table.txt` | Markdown table for this README |

---

## 📌 Project Overview

Online recruitment fraud is a growing problem, with fake job postings used to harvest personal data or money from job seekers. This project builds and compares multiple machine learning models to classify job postings as **real** or **fraudulent**, using both textual content (job description, requirements, company profile) and structured meta-features (company logo presence, telecommuting flag, employment type, etc.).

The project also performs **feature importance analysis** to identify which characteristics most strongly indicate a fraudulent posting — a key contribution beyond standard classification benchmarking.

---

## 🎯 Objectives

- Preprocess and engineer features from a real-world job postings dataset
- Handle severe class imbalance (~4.8% fraudulent postings)
- Train and compare multiple ML models for fraud classification
- Identify the most predictive features of fraudulent postings
- Present results in a format suitable for academic publication (IEEE/Springer)

---

## 📂 Dataset

**Source:** [Real / Fake Job Posting Prediction — Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)  
**Original source:** Employment Scam Aegean Dataset (EMSCAD), University of the Aegean

- **17,880** job listings
- **18** features (text + meta-data)
- Binary label: `fraudulent` (0 = real, 1 = fake)
- **866** fraudulent postings (**4.84%** of data)

File: `dataset/fake_job_postings.csv`

---

## 🧠 Methodology

### 1. Data Preprocessing
- **Text cleaning** (title, description, requirements, company_profile): lowercase, HTML tag removal, URL removal, special character removal, stopword removal (sklearn's ENGLISH_STOP_WORDS)
- **Missing value handling**: Fill categorical with 'Unknown', create binary indicator for missing salary
- **Text length meta-features**: title_length, desc_length, req_length, profile_length, total_text_length

### 2. Feature Engineering
- **Categorical encoding** (LabelEncoder): employment_type, required_experience, required_education, industry, function, department, country (extracted from location)
- **TF-IDF Vectorization**: 3,000 features, n-grams (1,2), min_df=2, max_df=0.95, sublinear_tf=True
- **Meta-features**: 7 categorical encoded + 4 binary + 5 text length = 16 features
- **Combined matrix**: TF-IDF (3,000) + scaled meta-features (16) = **3,016 features**

### 3. Class Imbalance Handling
- **SMOTE** oversampling on training data only
- Training: 13,611 real / 693 fraud → **13,611 / 13,611** (balanced)

### 4. Models Trained
| Model | Configuration |
|-------|---------------|
| **Naive Bayes** | MultinomialNB (α=0.1), TF-IDF only |
| **Linear SVM** | CalibratedClassifierCV + LinearSVC, class_weight='balanced' |
| **Random Forest** | 200 estimators, max_depth=20, class_weight='balanced' |
| **XGBoost** | 200 estimators, max_depth=6, scale_pos_weight=19.6 |
| **Neural Network** | MLP (128,64,32), early_stopping, max_iter=100 |

### 5. Evaluation
- Train/test split: 80/20, stratified
- Cross-validation: 5-fold StratifiedKFold (F1 scoring)
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Naive Bayes | 0.9114 | 0.3386 | 0.8728 | 0.4879 | 0.9700 |
| Linear SVM | 0.9857 | 0.9236 | 0.7688 | 0.8391 | 0.9883 |
| Random Forest | 0.9782 | 0.7811 | 0.7630 | 0.7719 | 0.9851 |
| XGBoost | 0.9360 | 0.4247 | 0.9133 | 0.5798 | 0.9784 |
| **Neural Network** | **0.9877** | **0.9108** | **0.8266** | **0.8667** | **0.9925** |

**Key Findings:**
- **Best Overall (F1-Score):** Neural Network (0.8667)
- **Best Precision:** Linear SVM (0.9236) — fewest false alarms
- **Best Recall:** XGBoost (0.9133) — catches most fraudulent postings
- **Best AUC-ROC:** Neural Network (0.9925) — best overall discrimination

---

## 🔍 Feature Importance (Random Forest)

### Top 10 Most Predictive Features
| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | has_company_logo | 0.0390 | Binary Meta-Feature |
| 2 | profile_length | 0.0373 | Text Length |
| 3 | total_text_length | 0.0166 | Text Length |
| 4 | required_education_encoded | 0.0146 | Categorical (Encoded) |
| 5 | tfidf_growing | 0.0111 | TF-IDF (Text) |
| 6 | has_questions | 0.0095 | Binary Meta-Feature |
| 7 | tfidf_work home | 0.0094 | TF-IDF (Text) |
| 8 | tfidf_earn | 0.0093 | TF-IDF (Text) |
| 9 | tfidf_data entry | 0.0090 | TF-IDF (Text) |
| 10 | country_encoded | 0.0085 | Categorical (Encoded) |

### Category Importance Summary
| Category | Total Importance | % of Total |
|----------|------------------|------------|
| TF-IDF (Text) | 0.847 | 84.7% |
| Text Length | 0.061 | 6.1% |
| Categorical (Encoded) | 0.058 | 5.8% |
| Binary Meta-Feature | 0.034 | 3.4% |

**Key Insight:** Missing company logo (`has_company_logo=0`) is the single strongest predictor of fraudulent postings, followed by short company profiles and overall text brevity. Text content (TF-IDF) dominates overall importance, with terms like "growing", "work home", "earn", "data entry" being highly indicative of fraud.

---

## 📁 Project Structure

```
FakeJobML/
├── dataset/
│   └── fake_job_postings.csv      # Raw dataset (50MB)
├── fake_job_detection.ipynb       # Main Jupyter notebook (37 cells)
├── run_pipeline_fast.py           # Standalone fast pipeline script
├── generate_feature_importance.py # Feature importance script
├── README.md                      # This file
├── results/                       # Generated outputs
│   ├── model_comparison.csv
│   ├── model_comparison_barchart.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance.csv
│   ├── feature_importance_top20.png
│   ├── feature_category_importance.png
│   └── readme_table.txt
├── .venv/                         # Virtual environment
└── mpl_config/                    # Matplotlib config (auto-created)
```

---

## 🛠️ Dependencies

All installed in `.venv`:
```
pandas==3.0.5
numpy==2.5.2
scikit-learn==1.9.0
xgboost==3.4.1
imbalanced-learn==0.14.2
nltk==3.10.3
matplotlib==3.11.1
seaborn==0.13.2
jupyter==1.1.1
```

---

## 📝 Research Paper / Manuscript

See `MANUSCRIPT.md` for a publication-ready manuscript draft.

---

## 📄 License

For academic research purposes. Dataset from Kaggle (EMSCAD).

---

## 👤 Author

**Ayan Bhaumik**  
📧 mrayanbhaumik@gmail.com / connect@ayanbhaumik.in

*Research project for Fake Job Posting Detection Using Machine Learning: A Comparative Study*
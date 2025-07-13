# ⚖️ Legal Document Analyser

This project is an **NLP + Machine Learning pipeline** that:
- Classifies legal clauses into 40+ categories (e.g., Parties, Governing Law, Renewal Term)
- Highlights key named entities (dates, organizations, amounts) using **spaCy**
- Visualizes EDA: clause type distribution, top TF-IDF terms, and confusion matrix
- Compares multiple models before and after **GridSearchCV hyperparameter tuning**
- Deploys an interactive **Streamlit** app for uploading and analysing contracts

---

## 📚 About

Manual contract review is tedious. This tool automates it by splitting text into clauses, predicting clause types, and extracting important entities to speed up legal document analysis.

---

## 🗃️ Dataset

- **Source:** CUAD (Contract Understanding Atticus Dataset)
- **Fields:**
  - `clause_text`: Raw text of the clause
  - `clause_type`: Label (e.g., Parties, Governing Law)

---

## ⚙️ How It Works

### 1️⃣ Training

- Runs EDA and saves visuals:
  - Clause type distribution
  - Top TF-IDF terms
  - Confusion matrix
- Trains **Logistic Regression**, **Random Forest**, **Linear SVC**, **Multinomial Naive Bayes**
- Tunes hyperparameters using **GridSearchCV**
- Compares model accuracies before and after tuning
- Saves the best pipeline as `best_pipeline.joblib` in `/models/`

### 2️⃣ Deployment

- The **Streamlit app** lets you upload a `.txt` contract
- It splits text into clauses, predicts each clause type, and highlights entities
- Results are shown in a clean table and with highlighted text

---

## 📊 Results Snapshot

| Model                        | Accuracy (Before)  | Accuracy (After Tuning) |
|------------------------------|--------------------|-------------------------|
| Logistic Regression          | ~0.7505            | ~0.7505                 |
| Random Forest                | ~0.687             | ~0.708                  |
| Linear SVC                   | ~0.742             | ~0.760                  |
| Multinomial Naive Bayes      | ~0.651             | ~                       |

Outputs:
- `outputs/clause_type_distribution.png`
- `outputs/model_comparison.png`
- `outputs/top_tfidf_terms.png`
- `outputs/confusion_matrix_[model].png`

---

## 🗂️ Project Structure

LegalDocumentAnalyser/
├── data/ # Cleaned CUAD dataset
├── models/ # Saved trained pipeline
├── outputs/ # EDA visuals & model comparison
├── Train_model.py # Training, EDA & tuning script
├── app.py # Streamlit web app


---

## 🔍 Key Insights

- Visual EDA shows clause distribution and feature importance
- Multiple models compared side by side with and without tuning
- GridSearchCV improves Logistic Regression & Linear SVC performance
- Final confusion matrix shows which clause types are most confusing

---

## ✨ Future Improvements

- Try legal domain-specific models like **InLegalBERT**
- Enhance clause splitting with better NLP parsing
- Add entity frequency summaries in the app

---

## 👨‍💻 Author

Sankarasetty Jaya Sri Ram
B.Tech CSE | Summer Internship Project

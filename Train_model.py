import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Create output folder
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load CUAD cleaned clauses dataset
df = pd.read_csv('cuad_clean.xls')
print(df.head())

# Clause Distribution
print("\nNumber of rows:", len(df))
print("\nColumns:", df.columns)

plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='clause_type', order=df['clause_type'].value_counts().index)
plt.title('Distribution of Clause Types')
plt.tight_layout()
plt.savefig("outputs/clause_type_distribution.png")
plt.show()

# Prepare Data
X = df['clause_text']
y = df['clause_type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

results = {}

# Logistic Regression Pipeline
pipeline_lr_default = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])
pipeline_lr_default.fit(X_train, y_train)
y_pred_lr_default = pipeline_lr_default.predict(X_test)
acc_lr_default = accuracy_score(y_test, y_pred_lr_default)
results['Logistic Regression'] = acc_lr_default
print(f"Logistic Regression : {acc_lr_default:.4f}")

# Logistic Regression Pipeline - GridSearchCV
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])
param_grid_lr = {
    'tfidf__max_features': [5000, 10000],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__max_iter': [100, 200, 500]
}
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, scoring='accuracy')
grid_lr.fit(X_train, y_train)
y_pred_lr = grid_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
results['Logistic Regression (After Tuning)'] = acc_lr
print("Best Logistic Regression Params:",grid_lr.best_params_)

# Random Forest Pipeline - BEFORE Tuning
pipeline_rf_default = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline_rf_default.fit(X_train, y_train)
y_pred_rf_default = pipeline_rf_default.predict(X_test)
acc_rf_default = accuracy_score(y_test, y_pred_rf_default)
results['Random Forest'] = acc_rf_default
print(f"Random Forest : {acc_rf_default:.4f}")

# Random Forest Pipeline - GridSearchCV
pipeline_rf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])
param_grid_rf = {
    'tfidf__max_features': [5000, 10000],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='accuracy')
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
results['Random Forest (After Tuning)'] = acc_rf
print("Best Random Forest Params: ",grid_rf.best_params_)


# Multinomial Naive Bayes Pipeline
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

pipeline_nb.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
results['Multinomial NB'] = acc_nb

# Linear SVC Pipeline - BEFORE Tuning
pipeline_svc_default = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LinearSVC())
])
pipeline_svc_default.fit(X_train, y_train)
y_pred_svc_default = pipeline_svc_default.predict(X_test)
acc_svc_default = accuracy_score(y_test, y_pred_svc_default)
results['Linear SVC'] = acc_svc_default
print(f"Linear SVC: {acc_svc_default:.4f}")

# Linear SVC Pipeline - GridSearchCV
pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])
param_grid_svc = {
    'tfidf__max_features': [5000, 10000],
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__max_iter': [1000, 2000]
}
grid_svc = GridSearchCV(pipeline_svc, param_grid_svc, cv=3, scoring='accuracy')
grid_svc.fit(X_train, y_train)
y_pred_svc = grid_svc.predict(X_test)
acc_svc = accuracy_score(y_test, y_pred_svc)
results['Linear SVC (After Tuning)'] = acc_svc
print("Best Linear SVC Params: ",grid_svc.best_params_)

# Model Comparison
print("Model Accuracies Comparison")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

# Bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Model Comparison (Accuracy)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()

# 6) Top TF-IDF Terms for Logistic Regression
tfidf = grid_lr.best_estimator_.named_steps['tfidf']
clf = grid_lr.best_estimator_.named_steps['clf']

feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_

if coefs.shape[0] > 1:
    coefs = np.mean(np.abs(coefs), axis=0)
else:
    coefs = coefs[0]

top_n = 20
top_indices = np.argsort(coefs)[-top_n:]
top_features = [feature_names[i] for i in top_indices]
top_importances = coefs[top_indices]

plt.figure(figsize=(12, 6))
sns.barplot(x=top_importances, y=top_features)
plt.title("Top TF-IDF Terms by Importance (Logistic Regression)")
plt.tight_layout()
plt.savefig("outputs/top_tfidf_terms.png")
plt.show()

# 7) Confusion Matrix for Best Model
from sklearn.metrics import confusion_matrix

best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name}")

# Select the best pipeline and predictions
if best_model_name == 'Logistic Regression (After Tuning)':
    final_pipeline = grid_lr.best_estimator_
    y_pred_best = y_pred_lr
    classes = grid_lr.classes_
elif best_model_name == 'Random Forest (After Tuning)':
    final_pipeline = grid_rf.best_estimator_
    y_pred_best = y_pred_rf
    classes = grid_rf.classes_
elif best_model_name == 'Multinomial NB':
    final_pipeline = pipeline_nb
    y_pred_best = y_pred_nb
    classes = pipeline_nb.classes_
else:
    final_pipeline = grid_svc.best_estimator_
    y_pred_best = y_pred_svc
    classes = grid_svc.classes_

# Compute confusion matrix and normalize
cm = confusion_matrix(y_test, y_pred_best, labels=classes)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot as heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(cm_normalized, cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True)
plt.title(f"Normalized Confusion Matrix Heatmap ({best_model_name})", fontsize=16)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"outputs/confusion_matrix_heatmap_{best_model_name.replace(' ', '_').lower()}.png")
plt.show()

# 8) Save Best Pipeline
joblib.dump(final_pipeline, 'models/best_pipeline.joblib')
print("\nFinal pipeline saved to 'models/best_pipeline.joblib'")

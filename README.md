# Syntecxhub_Spam_Detection
Project 2 - Week 2

# 📧 Spam Detection Project

## Project Overview
This project focuses on building a **machine learning model** to detect **spam messages** from text data. The workflow includes **data preprocessing, text vectorization, model training, evaluation, and saving the pipeline** for future predictions.

---

## Dataset
- Labeled dataset containing **spam and ham messages**.
- Columns: 
  - `label` → spam or ham  
  - `message` → text content of the message
- Common source: [SMS Spam Collection Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## Project Steps

1. **Data Preprocessing**
   - Convert text to lowercase.
   - Remove punctuation and numbers.
   - Tokenize text and clean it.
   - Optionally remove stopwords.

2. **Text Vectorization**
   - Use **TF-IDF Vectorizer** to convert text into numerical feature vectors.
   - Alternatively, **CountVectorizer** can be used.

3. **Model Training**
   - **Naive Bayes (MultinomialNB)** is used for classification.
   - Other options: Logistic Regression.

4. **Evaluation**
   - Metrics: **Accuracy, Precision, Recall, F1-score**
   - **Confusion Matrix** is used to visualize misclassifications.

5. **Saving the Model**
   - The full pipeline (**Vectorizer + Model**) is saved using `joblib` for future predictions.

6. **Prediction**
   - The saved pipeline can predict new messages easily.

---

## Requirements
- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

Install requirements using:

```bash
pip install pandas scikit-learn matplotlib joblib

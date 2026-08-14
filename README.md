Emotion Detection Using NLP

📌 Project Overview

This project focuses on Emotion Detection from Text using Natural Language Processing (NLP) and Machine Learning.

The model takes a text sentence as input and predicts the emotion expressed in that text. The project explores different text vectorization techniques and machine learning algorithms to identify the best-performing combination.

🎯 Objective

To build a machine learning model that can automatically classify text into its corresponding emotion category.

For example:

Input:  "I am very happy today!"
Output: Happy

📂 Dataset

The dataset contains two important columns:

- "text" – The input sentence/text.
- "emotion" – The target emotion label.

The emotion labels are converted into numerical values such as "0", "1", etc., for machine learning.

🔄 Project Workflow

Dataset
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Text Vectorization
   ↓
Machine Learning Models
   ↓
Prediction
   ↓
Accuracy Evaluation
   ↓
Model Comparison

🧹 Text Preprocessing

The text is processed before training the machine learning models. Preprocessing may include:

- Removing HTML tags
- Removing URLs
- Removing punctuation
- Converting text to lowercase
- Removing unnecessary characters
- Cleaning unwanted spaces

🔢 Text Vectorization

Two NLP vectorization techniques were explored:

1. Bag of Words

"CountVectorizer" converts text into numerical feature vectors based on word frequency.

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer()

X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

2. TF-IDF

"TfidfVectorizer" converts text into numerical features based on the importance of words.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

«The vectorizer is fitted only on the training data and then used to transform the test data to avoid data leakage.»

🤖 Machine Learning Models

The project experiments with models such as:

- Logistic Regression
- Multinomial Naive Bayes

The models are trained using the vectorized text data.

📊 Model Evaluation

The models are evaluated using Accuracy Score.

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

Different combinations of vectorizers and models are compared to identify the best-performing approach.

🏆 Best Model

In this project, Bag of Words + Logistic Regression gave the best performance among the tested approaches.

Therefore, this combination can be selected as the final model for emotion classification.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP
- Jupyter Notebook
- CountVectorizer
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes

📁 Project Structure

Emotion-Detection-NLP/
│
├── NLP.ipynb
├── dataset/
│   └── emotion_dataset.csv
│
└── README.md

🚀 Future Improvements

- Try advanced NLP preprocessing
- Test additional machine learning algorithms
- Perform hyperparameter tuning
- Use word embeddings such as Word2Vec
- Experiment with deep learning models
- Deploy the emotion detection model as a web application

👨‍💻 Conclusion

This project demonstrates how NLP and Machine Learning can be used to understand emotions expressed in text. By comparing different vectorization techniques and machine learning algorithms, the best-performing approach can be selected for emotion classification.

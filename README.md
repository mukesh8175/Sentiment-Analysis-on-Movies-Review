# Sentiment Analysis Using Machine Learning and NLP Techniques

This repository contains a project on sentiment analysis using machine learning and natural language processing (NLP) techniques. The aim of this project is to analyze textual data and determine the sentiment expressed, such as positive, negative, or neutral.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)

##Dataset  
             https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?select=IMDB+Dataset.csv
---

## Introduction

Sentiment analysis, also known as opinion mining, is one of the most prominent applications of Natural Language Processing (NLP) and machine learning. It involves analyzing and classifying text data to determine the sentiment or emotional tone expressed, such as positive, negative, or neutral. This technique is widely used across various domains, including customer feedback analysis, social media monitoring, brand perception studies, and market research, enabling organizations to gain valuable insights into customer opinions and preferences.

This project leverages advanced NLP techniques such as part-of-speech (POS) tagging for grammatical insights and Word2Vec for transforming text into meaningful numerical representations. Additionally, it employs machine learning models to effectively classify sentiments in text data, showcasing the integration of feature extraction methods with robust classification algorithms. The goal is to provide a comprehensive and accurate sentiment analysis pipeline that can be adapted to real-world scenarios.

---

## Features
- Data preprocessing using NLP techniques:
  - Tokenization
  - Stopword removal
  - Lemmatization
- Feature extraction:
  - POS tagging
  - Word2Vec (CBOW and Skip-gram models)
- Sentiment classification using machine learning algorithms
- Evaluation metrics for model performance

---

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - scikit-learn
  - NLTK
  - Gensim

---

## Project Workflow

1. **Data Collection:**
   - The dataset used for sentiment analysis was collected from reliable sources and loaded into the project for preprocessing.

2. **Data Preprocessing:**
   Text data needs to be cleaned and prepared before it can be used for machine learning models. This step involves the following NLP techniques:
   - **Tokenization:** Breaking text into individual words or tokens. For example, the sentence "I love programming!" is split into ["I", "love", "programming", "!"].
   - ## Data Preprocessing

- **Removing HTML Tags and Punctuation:** Cleaned the text by removing any HTML tags using regular expressions and eliminating punctuation marks to simplify the text data. For example: text = "<p>This is an example! Text with <b>HTML</b> tags.</p>". using **re.sub(r'<.*?>', '', text)**.
   - **Stopword Removal:** Removing common words like "the," "is," and "and" that do not add much value to the analysis. For instance, the sentence "I love the programming" becomes "I love programming."
   - **Lemmatization:** Reducing words to their base or root form while ensuring the meaning remains the same. For example, "running" becomes "run" and "better" becomes "good."
   - **Part-of-Speech (POS) Tagging:** Assigning grammatical tags to each word in the sentence, such as nouns, verbs, and adjectives. For instance, "I love programming" is tagged as [("I", "PRP"), ("love", "VB"), ("programming", "NN")].

3. **Feature Engineering:**
   After preprocessing, numerical representations of the text are created:
   - **Word2Vec:** Converts words into numerical vectors to capture semantic meanings. Two models are used:
     - **CBOW (Continuous Bag of Words):** Predicts a word based on its context.
     - **Skip-Gram:** Predicts the context given a word. This helps in capturing relationships between words.

4. **Model Training:**
   Three machine learning models were implemented and trained on the preprocessed data:
   - **Random Forest Classifier:**
     - A powerful ensemble learning method that combines multiple decision trees to improve classification accuracy.
     - Chosen for its ability to handle non-linear relationships and prevent overfitting through bagging.
   - **Naive Bayes Classifier:**
     - A probabilistic model based on Bayes' Theorem, particularly effective for text classification problems.
     - Chosen for its simplicity and efficiency in handling high-dimensional data.
   - **Logistic Regression:**
     - A linear model used for binary classification.
     - Chosen as a baseline model due to its simplicity and interpretability, providing insights into the relationships between features and sentiment labels.

5. **Evaluation:**
   - The models were evaluated using metrics like:
     - **Accuracy:** Measures the proportion of correct predictions.
     - **Precision:** Measures the proportion of true positive predictions out of all positive predictions.
     - **Recall:** Measures the proportion of true positive predictions out of actual positives.
     - **F1-Score:** Harmonic mean of precision and recall, balancing their trade-offs.
   - **Confusion Matrix:**
     - A confusion matrix was generated for each model to provide detailed insights into the classification performance.
     - The matrix represents counts of:
       - **True Positives (TP):** Correctly predicted positive sentiments.
       - **True Negatives (TN):** Correctly predicted negative sentiments.
       - **False Positives (FP):** Incorrectly predicted positive sentiments.
       - **False Negatives (FN):** Incorrectly predicted negative sentiments.
     - The confusion matrix helped identify specific areas where models misclassified sentiments, aiding in further optimization.

6. **Visualization:**
   - Results and insights were visualized using libraries like Matplotlib and Seaborn. Example visualizations include:
     - Sentiment distribution in the dataset.
     - Feature importance in Random Forest.
     - Confusion matrices for all models.
     - ROC curves for model comparisons.

7. **Result Analysis:**
   - The Random Forest model achieved the highest accuracy among the three, making it the most suitable for this dataset.
   - Naive Bayes performed well in cases of high-dimensional data and smaller feature sets.
   - Logistic Regression, as a baseline, provided valuable interpretability but was outperformed by the ensemble and probabilistic methods.
   - The confusion matrices revealed that Random Forest had fewer false positives and false negatives compared to the other models.


---

## Conclusion  

This project demonstrates the application of Natural Language Processing (NLP) and machine learning techniques for sentiment analysis. By employing methods like part-of-speech (POS) tagging, Word2Vec embeddings, and effective data cleaning techniques, the project builds a strong foundation for text preprocessing. Furthermore, the implementation of multiple classification models—Random Forest, Naive Bayes, and Logistic Regression—highlights their strengths in sentiment prediction.  

The Random Forest model emerged as the most accurate, showcasing its ability to handle complex data patterns, while Naive Bayes and Logistic Regression provided valuable insights into simpler relationships and baseline performance. Through rigorous evaluation using metrics like accuracy, precision, recall, F1-score, and confusion matrices, the project demonstrates the effectiveness of machine learning in real-world text classification tasks.  

This end-to-end pipeline serves as a robust framework for sentiment analysis, which can be extended or customized for various applications, such as analyzing product reviews, social media trends, or customer feedback. The insights gained from this project emphasize the importance of preprocessing, feature engineering, and model evaluation in delivering accurate and meaningful results.

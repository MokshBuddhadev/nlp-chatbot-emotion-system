# 🤖 NLP Chatbot with Emotion Detection using Machine Learning

## 📌 Project Overview

This project implements an intelligent chatbot using **Natural Language Processing (NLP)** and **Machine Learning**.
The chatbot can answer queries related to **AI / NLP concepts** and also detect the **emotion of the user** and respond accordingly.

The system uses dataset-based training, feature extraction, multiple machine learning algorithms, evaluation metrics, and a modern **Streamlit UI**.

This project was developed as part of an NLP / Machine Learning academic assignment and follows the required report format including dataset description, feature extraction, algorithms, hyperparameter tuning, and result analysis.

---

## 🚀 Features

* Intent based chatbot using ML
* Emotion detection chatbot
* NLP preprocessing pipeline
* TF-IDF feature extraction
* Bag of Words representation
* 3 Machine Learning algorithms

  * Naive Bayes
  * Support Vector Machine (SVM)
  * Logistic Regression
* Hyperparameter tuning using GridSearchCV
* Confusion matrix generation
* Accuracy / Precision / Recall / F1 evaluation
* Result visualization graphs
* Modern Streamlit chatbot UI
* Dataset based training
* Report ready outputs

---

## 📂 Project Structure

```
nlp_chatbot_project
│
├── data
│   ├── intents.csv
│   ├── emotions.csv
│   ├── responses.json
│
├── src
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_models.py
│   ├── chatbot.py
│   ├── emotion_model.py
│
├── ui
│   └── app.py
│
├── outputs
│   ├── NaiveBayes_cm.png
│   ├── SVM_cm.png
│   ├── Logistic_cm.png
│   ├── results.png
│
├── conversations
│   └── sample.txt
│
├── requirements.txt
├── main.py
└── README.md
```

---

## 📊 Dataset

Two datasets are used in this project:

### 1. intents.csv

Used for intent classification.

| text               | label        |
| ------------------ | ------------ |
| what is ai         | ai           |
| define nlp         | nlp          |
| applications of ai | applications |

### 2. emotions.csv

Used for emotion detection.

| text       | label   |
| ---------- | ------- |
| i am happy | happy   |
| i am sad   | sad     |
| i am angry | angry   |
| hello      | neutral |

Datasets are used to train machine learning models.

---

## ⚙️ NLP Preprocessing

The following preprocessing steps are applied:

* Lowercasing
* Tokenization
* Stopword removal
* Stemming
* Text cleaning

Libraries used:

* nltk
* sklearn

---

## 🔎 Feature Extraction

Two feature extraction methods are used:

### Bag of Words (BoW)

Converts text into word frequency vectors.

### TF-IDF

Term Frequency – Inverse Document Frequency.

Used for better performance.

Implemented using:

```
CountVectorizer
TfidfVectorizer
```

---

## 🤖 Machine Learning Algorithms

Three algorithms are used:

1. Naive Bayes
2. Support Vector Machine (SVM)
3. Logistic Regression

SVM gave best accuracy.

Hyperparameter tuning done using:

```
GridSearchCV
```

---

## 📈 Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Graphs generated using matplotlib.

Outputs saved in:

```
outputs/
```

---


## ▶️ How to Run Project

Install requirements:

```
pip install -r requirements.txt
```

Train models:

```
python -m src.train_models
```

Run chatbot (CLI):

```
python -m src.chatbot
```

Run emotion chatbot:

```
python -m src.emotion_model
```

Run Streamlit UI:

```
streamlit run ui/app.py
```

---

## 📷 Outputs

* Confusion Matrix images
* Result graphs
* Streamlit screenshots

Stored in:

```
outputs/
```

---

## 📌 Conclusion

This project successfully implements an NLP based chatbot with emotion detection using machine learning algorithms.

The system provides:

* Good accuracy
* Interactive UI
* Dataset based training
* Multiple ML models

The project satisfies all requirements of the assignment and demonstrates practical use of NLP and Machine Learning.

---

## 👨‍💻 Author

Moksh Buddhadev

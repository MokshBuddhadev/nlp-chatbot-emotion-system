import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.feature_extraction import load_dataset, tfidf_features


# -----------------------
# Load dataset
# -----------------------

df = load_dataset("data/intents.csv")

texts = df["clean"]
labels = df["label"]


# -----------------------
# Feature extraction (TFIDF improved)
# -----------------------

X, vectorizer = tfidf_features(texts)


# -----------------------
# Train test split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.3,
    random_state=42,
)


# -----------------------
# Hyperparameter tuning for SVM
# -----------------------

svm = LinearSVC()

params = {
    "C": [0.1, 1, 10]
}

grid = GridSearchCV(
    svm,
    params,
    cv=3
)

grid.fit(X_train, y_train)

best_svm = grid.best_estimator_

print("Best SVM:", best_svm)


# -----------------------
# Models
# -----------------------

models = {

    "NaiveBayes": MultinomialNB(),

    "SVM": best_svm,

    "Logistic": LogisticRegression(max_iter=1000),
}


results = {}


# -----------------------
# Training loop
# -----------------------

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    rec = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    results[name] = [acc, prec, rec, f1]

    print("\n", name)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)


    # -----------------------
    # Confusion Matrix
    # -----------------------

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot()

    plt.title(name)

    plt.savefig(f"outputs/{name}_cm.png")

    plt.close()


# -----------------------
# Result graph
# -----------------------

names = list(results.keys())

acc = [results[n][0] for n in names]
prec = [results[n][1] for n in names]
rec = [results[n][2] for n in names]
f1 = [results[n][3] for n in names]


x = range(len(names))


plt.figure()

plt.plot(x, acc, label="Accuracy")
plt.plot(x, prec, label="Precision")
plt.plot(x, rec, label="Recall")
plt.plot(x, f1, label="F1")

plt.xticks(x, names)

plt.legend()

plt.title("Model Comparison")

plt.savefig("outputs/results.png")

plt.show()
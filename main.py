# SPORTS VS POLITICS CLASSIFIER

import os
from data_loader import load_data

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt


# Evaluate Model

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return accuracy, precision, recall, f1, predictions


if __name__ == "__main__":

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # 1. Load Dataset
    X, y = load_data()
    print("Total samples:", len(X))

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 3. Define Feature Types
    feature_methods = {
        "Bag of Words": CountVectorizer(stop_words="english"),
        "TF-IDF": TfidfVectorizer(stop_words="english"),
        "TF-IDF (Unigram + Bigram)": TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    }

    # 4. Define Models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(max_iter=5000)
    }

    # Store results
    results = []

    # 5. Train & Evaluate
    for feature_name, vectorizer in feature_methods.items():

        # Convert text into numerical features
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        for model_name, model in models.items():

            acc, prec, rec, f1, preds = evaluate_model(
                model,
                X_train_vec,
                X_test_vec,
                y_train,
                y_test
            )

            results.append([
                feature_name,
                model_name,
                round(acc, 4),
                round(prec, 4),
                round(rec, 4),
                round(f1, 4)
            ])

            # Save confusion matrix for best model
            if feature_name == "TF-IDF" and model_name == "Linear SVM":
                cm = confusion_matrix(y_test, preds)
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=["Sport", "Politics"]
                )
                disp.plot()
                plt.title("Confusion Matrix - TF-IDF + Linear SVM")
                plt.savefig("results/confusion_matrix.png")
                plt.close()

    # 6. Printing Results
    print("\nComparison Results\n")
    print("{:<30} {:<20} {:<10} {:<10} {:<10} {:<10}".format(
        "Feature Type", "Model", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-" * 100)

    for row in results:
        print("{:<30} {:<20} {:<10} {:<10} {:<10} {:<10}".format(*row))

    # 7. Save Results
   
    with open("results/metrics.txt", "w") as f:

        header = "{:<30} {:<20} {:<10} {:<10} {:<10} {:<10}\n".format(
            "Feature Type", "Model", "Accuracy", "Precision", "Recall", "F1"
        )

        f.write(header)
        f.write("-" * 100 + "\n")

        for row in results:
            formatted_row = "{:<30} {:<20} {:<10} {:<10} {:<10} {:<10}\n".format(*row)
            f.write(formatted_row)


    print("\nResults saved to results folder.")
    

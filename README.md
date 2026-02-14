# Sports vs Politics Text Classification

## Overview

This project implements a binary text classification system that classifies news articles as either:

- Sport
- Politics

The goal is to compare different feature extraction techniques and machine learning models.

---

## Dataset

BBC News Dataset  
Available at: https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category  

Only two categories were selected:
- sport
- politics

Total samples: 928  
Training samples: 742  
Testing samples: 186  

A stratified 80/20 train-test split was used.

---

## Feature Representation

Three feature extraction techniques were compared:

1. Bag of Words  
2. TF-IDF  
3. TF-IDF (Unigram + Bigram)

---

## Machine Learning Models

The following models were evaluated:

- Naive Bayes  
- Logistic Regression  
- Linear Support Vector Machine (SVM)

---

## Results

TF-IDF based models achieved the highest performance.

Results and confusion matrix are saved in the `results` folder:

- `metrics.txt`
- `confusion_matrix.png`

---

## How to Run

Install dependencies:

```pip install -r requirements.txt```


Run the program:


```python src/main.py ```

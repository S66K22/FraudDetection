# 🚀 Maktab First Mini Project — Fraud Detection

## Problem Description

This project explores several machine learning models for **credit card fraud detection** and investigates how different preprocessing techniques, model choices, and evaluation metrics affect performance.

## Data Analysis

The dataset contains credit card transactions made by European cardholders in September 2013.

It contains:
- $284807$ samples
- $31$ columns which $30$ of them are `features` and last columns is `target` named `Class`.
- It has $1854$ duplicate rows which were removed in preprocessing step.
- Two preprocessed dataset are created. One is just created buy standard scaling and the other first was added two similarity time columns and removed time column.
- Dataset has two targets. 
  
```plain
0 → Legitimate transaction
1 → Fraudulent transaction
```

- Count of each targets are:

```plain
0    284315
1       492
```

---

## 🎯 Guiding Questions

### 1. Which model do you expect to perform best for fraud detection? Why?

**Random Forest**.

I expect Random Forest to perform well because it combines multiple decision trees, where each tree is trained using a subset of the data and features. This ensemble approach can make the model more robust and better at capturing complex patterns in fraud transactions.

It can also handle non-linear relationships and, with appropriate `class_weight` settings, can be adapted to imbalanced classification problems.

---

### 2. Which metric is more important for this problem: Precision, Recall, or F1-score? Why?

For this particular problem, I consider **Recall** to be the most important metric.

In fraud detection, we want to minimize the number of fraudulent transactions that the model fails to detect. In other words, we don't want fraudulent transactions to **escape detection**.

A false positive (legitimate transaction classified as fraud) can be investigated further, while a false negative (fraud classified as legitimate) means the fraudulent transaction may go completely undetected.

Therefore, I would prioritize **high Recall**, while still monitoring Precision to make sure the number of false alarms remains reasonable.

> **Goal:** Detect as many fraudulent transactions as possible while keeping false positives manageable.

---

### 3. What do you expect to happen if the model predicts all transactions as legitimate?

If the model predicts every transaction as legitimate:

- **Accuracy** may appear very high because fraud transactions are extremely rare.
- **Precision** becomes undefined or effectively unusable because the model never predicts the positive class.
- **Recall** becomes **0**, because the model detects none of the fraudulent transactions.
- **F1-score** also becomes **0**.

Therefore, a very high accuracy in this situation would be misleading.

This demonstrates why **accuracy alone is not a suitable metric for highly imbalanced fraud detection problems**.

---

### 4. Do you expect feature scaling to significantly affect KNN performance?

**Yes.**

KNN relies heavily on **distance calculations** to determine which observations are similar to each other.

If features have significantly different scales, features with larger numerical ranges can dominate the distance calculation.

For example:

```text
Feature A: 0.0  → 1.0
Feature B: 0   → 100,000

Explain in your README.md:

Was your initial hypothesis correct?
Which model performed best?
Which metric was most informative?
How did class imbalance affect the results?
What was the trade-off between False Positives and False Negatives?
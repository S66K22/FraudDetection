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

I expect Deep Model to perform well because it has great capability of learning non-linear patters and learn more complex patters.

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
```

---

### 5. Was your initial hypothesis correct?
My initial hypothesis was partially correct. I expected the Deep Model to perform well because deep neural networks can learn complex non-linear relationships between features. This expectation was supported by the results, as the Deep Model achieved the best overall performance according to PR-AUC. However, Logistic Regression also performed surprisingly well and achieved a higher Recall while requiring significantly less training time and memory

---

### 6. Which model performed best?
The Deep Model performed best according to PR-AUC, making it the strongest model for overall fraud detection performance in this experiment. However, Logistic Regression achieved higher Recall and required significantly less computational resources. Therefore, the Deep Model was the best-performing model according to the selected metric, while Logistic Regression may be a more practical choice when computational efficiency and fraud detection Recall are prioritized.

---

### 7. Which metric was most informative?
PR-AUC was the most informative metric. Since fraudulent transactions represent only a very small fraction of all transactions, ROC-AUC and especially Accuracy can sometimes give an overly optimistic impression of model performance. PR-AUC focuses more directly on the model's ability to identify the positive class while maintaining reasonable precision.

---

### 8. How did class imbalance affect the results?
Class imbalance strongly affected the results. Fraudulent transactions represent only a very small portion of the dataset. If the classes are treated equally in the loss function, the large number of legitimate transactions has a much greater influence on the model's learning process. As a result, a model can achieve very high accuracy by mostly predicting transactions as legitimate while still performing poorly at detecting fraud. This is why metrics such as Recall, F1-score, and particularly PR-AUC are more informative for this problem.

---

### 9. What was the trade-off between False Positives and False Negatives?
There is a trade-off between False Positives and False Negatives. Improving the detection of fraudulent transactions can increase False Positives, while reducing False Positives can lead to more fraudulent transactions being missed. Therefore, we need to find a suitable balance between detecting fraud and avoiding false alarms.
Absolutely. I would make the report more structured by clearly separating **experimental setup**, **baseline results**, **feature-engineered results**, and **final model selection**. I’d also highlight the improvement from the Isolation Forest features rather than just listing the numbers.

# Deep Learning Model Experiments

## 1. Experimental Setup

Several feed-forward neural network architectures were evaluated for the fraud detection task.

The primary evaluation metric is **PR-AUC (Precision-Recall Area Under the Curve)** because the fraud detection dataset is highly imbalanced. PR-AUC provides a more informative measure of performance than accuracy when the positive class (fraud) is rare.

Each model was evaluated using **5-fold cross-validation**. For each fold, the **best validation PR-AUC** achieved during training was recorded.

The reported `Mean PR-AUC` is the average of the best validation PR-AUC values across the five folds.

### Model Architectures

All evaluated models contain **two hidden layers**. The hidden-layer dimensions were varied to compare different model capacities:

* `[32, 32]`
* `[64, 32]`
* `[64, 64]`
* `[128, 64]`
* `[128, 128]`

The input dataset was **standard-scaled** before training.

---

# 2. Experiments with the Original Features

The first set of experiments used the original feature set after standard scaling.

| Architecture | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean PR-AUC |
| ------------ | -----: | -----: | -----: | -----: | -----: | ----------: |
| `[32, 32]`   | 0.8501 | 0.8752 | 0.7402 | 0.8756 | 0.8827 |  **0.8448** |
| `[64, 32]`   | 0.8451 | 0.8701 | 0.7100 | 0.7713 | 0.8836 |  **0.8160** |
| `[64, 64]`   | 0.8580 | 0.7948 | 0.7655 | 0.8182 | 0.8752 |  **0.8223** |
| `[128, 64]`  | 0.8564 | 0.8767 | 0.7709 | 0.7909 | 0.8949 |  **0.8380** |
| `[128, 128]` | 0.8579 | 0.8626 | 0.7676 | 0.8847 | 0.8924 |  **0.8530** |

### Best Model on Original Features

The best architecture was:

**`[128, 128]` — Mean PR-AUC: 0.8530**

This model achieved the highest average PR-AUC among the architectures evaluated using the original features.

---

# 3. Experiments with Isolation Forest Features

Additional features derived from **Isolation Forest** were then added to the dataset.

These features were intended to provide the neural network with additional information about the degree to which an observation resembles an anomalous or unusual sample.

The same five neural network architectures were evaluated again.

| Architecture | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean PR-AUC |
| ------------ | -----: | -----: | -----: | -----: | -----: | ----------: |
| `[32, 32]`   | 0.8551 | 0.8656 | 0.7772 | 0.8121 | 0.8892 |  **0.8398** |
| `[64, 32]`   | 0.8567 | 0.7942 | 0.7653 | 0.7961 | 0.8349 |  **0.8094** |
| `[64, 64]`   | 0.8494 | 0.8696 | 0.7601 | 0.8788 | 0.8829 |  **0.8481** |
| `[128, 64]`  | 0.8539 | 0.8766 | 0.7703 | 0.8846 | 0.8973 |  **0.8565** |
| `[128, 128]` | 0.8554 | 0.8786 | 0.7733 | 0.8845 | 0.8975 |  **0.8579** |

### Best Model with Isolation Forest Features

The best architecture was:

**`[128, 128]` — Mean PR-AUC: 0.8579**

---

# 4. Comparison of the Two Feature Sets

The best architecture was `[128, 128]` for both feature sets.

| Feature Set                          | Best Architecture | Mean PR-AUC |
| ------------------------------------ | ----------------- | ----------: |
| Original features                    | `[128, 128]`      |      0.8530 |
| Original + Isolation Forest features | `[128, 128]`      |  **0.8579** |

Adding the Isolation Forest-derived features improved the mean PR-AUC from:

**0.8530 → 0.8579**

This represents an absolute improvement of approximately **0.0049 PR-AUC points**.

The improvement is relatively small, but it suggests that the additional anomaly-related information provided by the Isolation Forest features may contain useful information for fraud detection.

---

# 5. Model Selection

Based on the 5-fold cross-validation results, the selected deep learning architecture is:

### **Two hidden layers: `[128, 128]`**

using the dataset containing the **additional Isolation Forest-derived features**.

Its mean validation PR-AUC was:

### **0.8579**

This was the highest mean PR-AUC observed among all evaluated deep learning configurations.

The selected architecture will therefore be used for the next stage of the modeling pipeline, where the model can be retrained using the complete training dataset before evaluating its final performance on the held-out test set.

---

# 6. Key Observations

Several observations can be made from these experiments:

1. **The `[128, 128]` architecture performed best overall.**
   It achieved the highest mean PR-AUC for both feature sets.

2. **Increasing model capacity does not always improve performance.**
   For example, `[64, 32]` performed substantially worse than `[128, 128]`. This suggests that the relationship between network size and performance is not simply monotonic.

3. **The Isolation Forest features provided a small improvement for the best architecture.**
   The `[128, 128]` model improved from **0.8530 to 0.8579** after adding the additional features.

4. **There is noticeable variation between folds.**
   For example, the selected `[128, 128]` model with Isolation Forest features achieved PR-AUC values ranging from **0.7733 to 0.8975**. This indicates that model performance depends considerably on the particular validation fold.

5. **PR-AUC is the appropriate primary metric for this experiment.**
   Because fraud detection is an imbalanced classification problem, PR-AUC focuses more directly on the model's ability to identify fraudulent transactions while maintaining useful precision.

---

## 7. Final Result

**Selected model:**

```text
Architecture:       [128, 128]
Feature set:        Original + Isolation Forest features
Cross-validation:   5-fold
Mean validation
PR-AUC:             0.8579
```

This model is currently the best-performing deep learning configuration based on cross-validation.

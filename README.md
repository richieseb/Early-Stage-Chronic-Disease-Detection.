**# Comparative Analysis of Probabilistic and Distance-Based Classifiers for Early-Stage Chronic Disease Detection

An academic engineering framework designed to benchmark **Gaussian Naïve Bayes (GNB)** and **K-Nearest Neighbors (KNN)** algorithms using clinical healthcare analytics. This project explores the mathematical trade-offs between instance-based geometric spaces and parametric probabilistic modeling to assist in early-stage diagnostic decision support systems.

---
## 📌 Project Overview
This project presents a comparative engineering analysis between two fundamentally distinct machine learning paradigms—**Probabilistic Decision Theory** and **Geometric Instance-Based Learning**—applied to the critical domain of healthcare diagnostics. 

Using patient clinical metrics (such as Blood Pressure and Serum Creatinine), we benchmark **Gaussian Naïve Bayes (GNB)** against an optimized **K-Nearest Neighbors (KNN)** classifier to predict early-stage Chronic Kidney Disease (CKD). The core objective is to evaluate how feature scaling and hyperparameter tuning influence classification boundaries under strict medical constraints.

---

## 🎯 Research Objectives
1. **Data Preprocessing & Engineering:** Implement robust missing data handling via median/mode imputation and resolve structural feature imbalances using Min-Max Normalization.
2. **Hyperparameter Optimization:** Conduct a systematic cross-validation sweep to isolate the optimal neighborhood parameter ($K$) for instance-based classification, mitigating overfitting risks.
3. **Comparative Performance Benchmarking:** Appraise both architectures using multi-class evaluation metrics including Classification Accuracy, Precision, Recall (Sensitivity), F1-Score, and Confusion Matrices.

---

## 🔬 Core Mathematical Frameworks

### 1. Gaussian Naïve Bayes (GNB)
GNB handles continuous clinical features by assuming they fit a normal Gaussian distribution within each class. It relies on Bayes' Theorem, operating under a strict assumption of feature conditional independence:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)$$

Where:
* $\mu_y$ is the calculated mean of feature $x_i$ for class $y$.
* $\sigma_y^2$ is the variance of feature $x_i$ for class $y$.

### 2. K-Nearest Neighbors (KNN)
KNN maps patient records directly into a multi-dimensional metric space. It calculates the proximity of an unclassified instance to historical cases using the Euclidean Distance metric:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

The model dynamically assigns a diagnostic label based on a majority vote of its $K$ closest neighbors.

---

# Titanic Passenger Survival Prediction Pipeline

An end-to-end Machine Learning classification pipeline built with **Python**, **Pandas**, and **Scikit-Learn**. This repository demonstrates binary classification using an ensemble **Random Forest** algorithm, highlighting key ML engineering concepts including feature encoding, missing value imputation, and model evaluation on unseen test data.

---

## 📌 Project Architecture & Technical Highlights

This project implements a structured Machine Learning pipeline to predict binary survival outcomes (`1` = Survived, `0` = Did Not Survive) based on passenger demographic and ticket metadata.

Key engineering decisions implemented in this script:

### 1. Feature Encoding & Impact of the `Sex` Feature
Machine Learning models rely on mathematical matrix operations and cannot natively process raw string categories (e.g., `"male"`, `"female"`).
* **Implementation:** Mapped categorical string values to binary numerical indicators (`female` $\rightarrow$ `1`, `male` $\rightarrow$ `0`).
* **Impact on Model Performance:** On the Titanic, historical survival rates heavily favored female passengers due to the "women and children first" evacuation protocol. Incorporating the encoded `Sex` feature improved baseline model accuracy from **~67%** (using only `Pclass`, `Fare`, `Age`) to **~78%+**, proving to be the single most influential predictive feature in the dataset.

### 2. Validation Strategy: 80/20 Train-Test Split
To measure true model generalization and prevent **overfitting** (memorizing training patterns without learning real trends), the dataset is split into two disjoint sets:
* **Training Set (80%):** Passed to `model.fit(X_train, y_train)` to construct internal decision boundaries.
* **Testing Set (20%):** Kept completely isolated during training. Evaluated via `model.predict(X_test)` to simulate real-world inference on unseen data.
* **Reproducibility:** Configured with a static `random_state=42` seed to ensure deterministic, repeatable validation splits across runs.

### 3. Ensemble Model Dynamics: `RandomForestClassifier`
Instead of relying on a single Decision Tree (which is prone to high variance and overfitting), this project utilizes a **Random Forest Classifier**:
* **Decision Tree Ensembles:** Constructs 100 parallel decision trees (`n_estimators=100`), where each tree is trained on a random bootstrap sample of the dataset.
* **Feature Interaction Evaluation:** Trees split nodes by evaluating non-linear feature interactions (e.g., identifying that *high ticket fare combined with 1st class status* significantly increases survival probability).
* **Majority Voting:** Final predictions represent the aggregated majority vote across all individual decision trees, delivering lower variance and higher stability.

---

## 🛠️️ Project Structure

```text
├── data/
│   └── titanic.csv               # Raw dataset
|── Titanic-Binary-Classification-Pipeline.py         # End-to-end data loading, encoding, training & evaluation
├── requirements.txt              # Project dependencies
└── README.md                     # Technical documentation

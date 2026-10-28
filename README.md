# NEURODIAB — ANN-Based Diabetes Risk Prediction System

An Artificial Neural Network (ANN)-based machine learning project for predicting diabetes risk using symptom-based questionnaire data.

## 📌 Project Overview

**NEURODIAB** is a machine learning project that explores the use of Artificial Neural Networks to classify potential diabetes risk based on symptoms and demographic information. The project uses the **UCI Early Stage Diabetes Risk Prediction Dataset** and applies data preprocessing, categorical encoding, feature scaling, model training, and performance evaluation.

The objective is to develop a data-driven prediction system that can support preliminary risk assessment for educational and research purposes.

> **Disclaimer:** NEURODIAB is an academic prototype, not a clinically validated diagnostic tool. Its predictions must not replace professional medical advice, laboratory testing, or diagnosis by a qualified healthcare professional.

## 🎯 Objectives

- Develop an ANN-based diabetes risk classification model.
- Preprocess numerical and categorical questionnaire features.
- Evaluate the model using classification and probability-based metrics.
- Examine model performance using a held-out test set and grouped cross-validation.
- Prepare a foundation for comparing alternative machine learning models.
- Plan a user-friendly interface for future prediction workflows.

## 📊 Dataset

**Dataset:** UCI Early Stage Diabetes Risk Prediction Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)

| Property | Details |
|---|---|
| Total records | 520 |
| Input features | 16 |
| Target variable | `class` |
| Positive class | 320 records |
| Negative class | 200 records |
| Missing cells | 0 |
| Data type | Symptom-based questionnaire data |

The dataset includes demographic information and symptoms such as increased thirst, frequent urination, sudden weight loss, weakness, and other indicators associated with early-stage diabetes risk.

The dataset is questionnaire-based and should not be interpreted as a collection of laboratory-confirmed clinical measurements.

## 🧠 Methodology

The project follows these stages:

1. **Data collection:** Load the UCI Early Stage Diabetes Risk Prediction Dataset.
2. **Data inspection:** Examine feature types, target distribution, missing values, and duplicate records.
3. **Preprocessing:**
   - Numerical features: median imputation and standardization.
   - Categorical features: most-frequent imputation and one-hot encoding.
4. **Model development:** Train an Artificial Neural Network using `MLPClassifier`.
5. **Evaluation:** Measure classification performance on a held-out test set.
6. **Cross-validation:** Use five-fold stratified grouped cross-validation to reduce the risk of exact duplicate records appearing across different folds.
7. **Future development:** Compare alternative classifiers and develop a prediction interface.

### ANN Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | Artificial Neural Network |
| Implementation | Scikit-learn `MLPClassifier` |
| Hidden layers | 32 and 16 neurons |
| Activation function | ReLU |
| Optimizer | Adam |
| Regularization | L2, alpha = 0.001 |
| Initial learning rate | 0.001 |
| Maximum iterations | 1000 |
| Early stopping | Enabled |
| Random state | 42 |

The preprocessing pipeline and ANN are trained together to help prevent preprocessing leakage from the test set.

## 📈 Initial ANN Baseline Results

The following results are from the initial held-out test evaluation.

| Metric | Result |
|---|---:|
| Accuracy | 83.65% |
| Precision | 81.33% |
| Recall / Sensitivity | 95.31% |
| Specificity | 65.00% |
| F1-score | 87.77% |
| ROC-AUC | 0.9707 |
| PR-AUC (Average Precision) | 0.9838 |
| Brier score | 0.0945 |

### Five-Fold Grouped Cross-Validation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 90.58% | 5.98% |
| Precision | 92.38% | 6.28% |
| Recall | 92.81% | 7.38% |
| F1-score | 92.36% | 4.89% |
| ROC-AUC | 0.9658 | 0.0283 |

**Interpretation:** The initial ANN baseline achieved high recall, meaning it identified a large proportion of positive cases in the held-out test set. Its specificity was lower, indicating that some negative cases were incorrectly classified as positive.

These are preliminary experimental results. The dataset is relatively small, and external validation on independent clinical data has not been performed. Cross-validation results should be interpreted separately from the held-out test results.

## 🛠️ Technologies Used

- **Programming language:** Python
- **Machine learning:** Scikit-learn
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Development environment:** Visual Studio Code
- **Version control:** Git and GitHub

## 📁 Repository Structure

```text
Diabetes-Prediction-System/
├── .vscode/
└── NeuroDiab_Review1_Code_Graphs/
    ├── review2_results/
    │   └── reports/
    │       ├── run_summary.json
    │       └── test_predictions.csv
    └── reports/
        ├── heldout_predictions.csv
        └── model_comparison_metrics.csv
```

*Note:* This structure reflects the currently identified repository files. Additional training scripts, model files, datasets, and evaluation visualizations should be added to the README when they are committed to the repository.

## 🚀 Installation and Execution

### Prerequisites

- Python 3.10 or a compatible version
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/RRRuban/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### 4. Run the project

The committed repository should include the corresponding training or prediction script before an execution command is provided. Once that script is committed, add its exact filename and command here.

## 🔬 Future Enhancements

- Compare ANN performance against Logistic Regression, Decision Tree, Random Forest, SVM, and KNN.
- Perform additional model validation and threshold analysis.
- Investigate class imbalance and false-positive/false-negative trade-offs.
- Add explainability techniques to improve model interpretability.
- Develop a Streamlit-based user interface.
- Validate performance on suitable independent datasets before considering real-world use.

These are planned enhancements and should not be considered completed unless implemented and evaluated.

## 👥 Project Team

**Project:** NEURODIAB — ANN-Based Diabetes Risk Prediction System

**Institution:** SRM Institute of Science and Technology, Tiruchirappalli

**Department:** Computer Science and Engineering — Artificial Intelligence and Machine Learning

**Team Members:**
- Jaya Ruban P
- Anbu Chelvan KR

**Project Guide:** Dr. Bhuvaneshwari Amma, ASP/AIML

## 📚 References

1. UCI Machine Learning Repository, *Early Stage Diabetes Risk Prediction Dataset*. [Dataset page](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset).
2. Scikit-learn Developers, *MLPClassifier Documentation*. [Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).
3. Scikit-learn Developers, *Model Evaluation*. [Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html).
4. Scikit-learn Developers, *Pipelines and Composite Estimators*. [Documentation](https://scikit-learn.org/stable/modules/compose.html).

## ⚠️ Limitations and Disclaimer

- The dataset contains only 520 records.
- Questionnaire responses may not represent laboratory-confirmed diagnoses.
- Performance may vary across populations and data collection settings.
- The initial results have not been externally or clinically validated.
- The system is intended for academic research and educational demonstration only.

**NEURODIAB must not be used as a standalone medical diagnostic system.**

## 📄 License

No license has been specified yet. Add an appropriate open-source license if you intend to permit others to reuse, modify, or distribute this project.

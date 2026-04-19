# 🌸 Iris Flower Classification
### CodeAlpha Data Science Internship — Task 1

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project builds a machine learning classification system to identify **three species of Iris flowers** — *Setosa*, *Versicolor*, and *Virginica* — based on their sepal and petal measurements. It demonstrates end-to-end ML workflow including data exploration, preprocessing, model training, and evaluation.

---

## 🗂 Dataset

- **Source:** UCI Machine Learning Repository (via `sklearn.datasets.load_iris`)
- **Samples:** 150 (50 per class)
- **Features:** 4 numerical features
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Target Classes:** Setosa | Versicolor | Virginica

---

## 🧪 Models Trained

| Model | Accuracy |
|---|---|
| K-Nearest Neighbors | 93.33% |
| Decision Tree | 93.33% |
| Random Forest | 90.00% |
| **Support Vector Machine** ✅ | **96.67%** |

> ✅ Best performing model: **Support Vector Machine (SVM)** with RBF kernel

---

## 📊 Visualizations Generated

| File | Description |
|---|---|
| `iris_eda.png` | Scatter plots of all feature pairs colored by species |
| `iris_correlation.png` | Heatmap of feature correlations |
| `iris_boxplot.png` | Boxplots showing feature distributions per species |
| `iris_model_results.png` | Confusion matrix + accuracy comparison bar chart |
| `iris_feature_importance.png` | Feature importance from Random Forest |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
```

### 2. Install Dependencies
```bash
pip install scikit-learn pandas matplotlib seaborn numpy
```

### 3. Run the Script
```bash
python task1_iris_classification.py
```

---

## 📁 Project Structure

```
CodeAlpha_IrisFlowerClassification/
│
├── task1_iris_classification.py   # Main Python script
├── iris_eda.png                   # EDA scatter plots
├── iris_correlation.png           # Correlation heatmap
├── iris_boxplot.png               # Feature boxplots
├── iris_model_results.png         # Model results & confusion matrix
├── iris_feature_importance.png    # Feature importance chart
└── README.md                      # Project documentation
```

---

## 🔍 Key Findings

- **Petal length and petal width** are the most discriminative features for classification
- *Setosa* is perfectly linearly separable from the other two species
- SVM with RBF kernel achieved the highest accuracy of **96.67%**
- All models achieved above 90% accuracy, demonstrating the dataset is well-structured

---

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## 👤 Author

**[Priyadarshini Lodh]**
CodeAlpha Data Science Intern


---

*This project was completed as part of the CodeAlpha Data Science Internship Program.*

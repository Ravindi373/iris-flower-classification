# Iris Flower Classification

A machine learning pipeline built with **scikit-learn** to classify Iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on sepal and petal measurements.

## Objective

Classify Iris flowers into 3 species based on 4 measurements, using and comparing multiple ML models wrapped in a clean `scikit-learn` Pipeline (no data leakage, easy to deploy).

##  Dataset

150 samples, 4 features, 3 balanced classes (50 samples each), no missing values:

| Feature | Description |
|---|---|
| `SepalLengthCm` | Length of the sepal (cm) |
| `SepalWidthCm` | Width of the sepal (cm) |
| `PetalLengthCm` | Length of the petal (cm) |
| `PetalWidthCm` | Width of the petal (cm) |
| `Species` | Target label — Iris-setosa / Iris-versicolor / Iris-virginica |

## 🔍 Exploratory Data Analysis

- Pairplot, correlation heatmap, and boxplots by species to visualize feature separation
- **Petal measurements separate the species far better than sepal measurements**

## Pipeline

Data is split 80/20 (train/test, stratified, `random_state=42`), and each model is wrapped in a pipeline of `StandardScaler` → classifier:

- **Logistic Regression** (`max_iter=200`)
- **Support Vector Machine** (RBF kernel)
- **Random Forest** (100 estimators)

Each model is evaluated with train accuracy, test accuracy, and 5-fold cross-validation.

## Results

| Model | Train Acc | Test Acc | CV Mean | CV Std |
|---|---|---|---|---|
| Logistic Regression | 0.9583 | 0.9333 | 0.9583 | 0.0264 |
| **Support Vector Machine** | 0.9750 | **0.9667** | **0.9667** | 0.0312 |
| Random Forest | 1.0000 | 0.9000 | 0.9500 | 0.0167 |

** Best model: Support Vector Machine** (CV mean accuracy 96.67%)

Classification report (SVM, test set):

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Iris-setosa | 1.00 | 1.00 | 1.00 |
| Iris-versicolor | 1.00 | 0.90 | 0.95 |
| Iris-virginica | 0.91 | 1.00 | 0.95 |

Overall test accuracy: **97%**

## Feature Importance (Random Forest)

| Feature | Importance |
|---|---|
| PetalWidthCm | 0.4372 |
| PetalLengthCm | 0.4315 |
| SepalLengthCm | 0.1163 |
| SepalWidthCm | 0.0150 |

Petal measurements dominate — sepal width contributes almost nothing to classification.

## Sample Predictions

```
SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  Predicted Species
          5.1           3.5            1.4           0.2       Iris-setosa
          6.0           2.7            5.1           1.6    Iris-virginica
          6.9           3.1            5.4           2.1    Iris-virginica
```

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn (Pipeline, StandardScaler, LogisticRegression, SVC, RandomForestClassifier)
- Matplotlib, Seaborn

## Project Structure

```
iris-flower-classification/
├── Iris.csv                          # Dataset
├── Iris_Flower_Classification.ipynb  # Full pipeline: EDA, training, evaluation, predictions
└── README.md
```

##  How to Run

### Option 1: Google Colab (recommended)
Open `Iris_Flower_Classification.ipynb` in [Google Colab](https://colab.research.google.com/) and run all cells (you'll be prompted to upload `Iris.csv`).

### Option 2: Run locally
```bash
git clone https://github.com/Ravindi373/iris-flower-classification.git
cd iris-flower-classification
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Iris_Flower_Classification.ipynb
```

## Key Findings

- `PetalLengthCm` and `PetalWidthCm` are by far the most important features for classification.
- All three models achieve high accuracy (~90–100%) because the Iris dataset is small, balanced, and well-separated.
- Wrapping preprocessing + model training in a single `Pipeline` keeps the workflow clean and reproducible.

## Author

**Ravindi Ayodhya**

## License

This project is open source and available under the [MIT License](LICENSE).

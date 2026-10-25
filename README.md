# Iris Species Classification

A machine-learning project that classifies Iris flowers into **Setosa, Versicolor, or Virginica** using their sepal and petal measurements. The notebook explores traditional classification models alongside hyperparameter tuning and ensemble learning.

## Project Overview

This project uses the classic Iris dataset available through Seaborn and Scikit-learn. It demonstrates a supervised machine-learning workflow: loading the data, separating features from the target, splitting data into training and testing sets, fitting classifiers, and comparing model performance.

## Dataset

The notebook loads the data with:

```python
df = sns.load_dataset("iris")
```

The dataset contains four numeric input features and one target column:

| Column | Description |
|---|---|
| `sepal_length` | Sepal length of the flower |
| `sepal_width` | Sepal width of the flower |
| `petal_length` | Petal length of the flower |
| `petal_width` | Petal width of the flower |
| `species` | Iris species to predict |

The three target classes are `setosa`, `versicolor`, and `virginica`.

## What the Notebook Covers

- Load and preview the Iris dataset.
- Identify the available species.
- Separate the measurements (`X`) from the target label (`y`).
- Split the data into training and test sets.
- Train a K-Nearest Neighbors (KNN) classifier.
- Train a Support Vector Classifier (SVC).
- Tune SVC hyperparameters using `GridSearchCV` and `RandomizedSearchCV`.
- Build a stacking ensemble with Decision Tree, SVC, and Logistic Regression base learners.
- Train and evaluate a Random Forest classifier.

## Models and Recorded Results

The saved notebook outputs report the following results for the initial 80/20 train-test split (`random_state=42`):

| Model | Recorded test accuracy |
|---|---:|
| K-Nearest Neighbors (KNN, `n_neighbors=13`) | 100% |
| Support Vector Classifier (SVC, `gamma='auto'`) | 100% |

The notebook also explores SVC hyperparameter tuning with 5-fold cross-validation. In the saved Grid Search results, the highest displayed mean cross-validation score is approximately **95.83%**. Results may vary when the notebook is rerun with different library versions or settings.

## Tech Stack

- **Python**
- **NumPy** and **Pandas** — data handling
- **Matplotlib** and **Seaborn** — visualization and dataset loading
- **Scikit-learn** — classification, model selection, hyperparameter tuning, and ensemble learning
- **Jupyter Notebook / Google Colab** — interactive development

## Run the Project

1. Clone this repository:

   ```bash
   git clone https://github.com/1kunal2006/Iris-project.git
   cd Iris-project
   ```

2. Install the required libraries:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

3. Open `iris.ipynb` in Jupyter Notebook, or upload it to Google Colab.
4. Run the cells in order.

The notebook loads the dataset from Seaborn, so you do not need to download a separate CSV file.

## Learning Outcomes

By exploring this project, you can practise:

- Building classification models from tabular data.
- Comparing different machine-learning algorithms.
- Searching for useful model hyperparameters.
- Combining models through stacking.
- Interpreting held-out test accuracy and cross-validation scores.

## Author

Kunal

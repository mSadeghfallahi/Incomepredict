# Income Level Prediction with UCI Adult Dataset

A comprehensive machine learning project to predict whether an individual's annual income exceeds $50,000. This repository covers the entire pipeline from exploratory data analysis and feature engineering to model comparison and hyperparameter tuning.

## Project Overview

The goal of this project is to tackle a classic binary classification problem using the well-known **UCI Adult census dataset**. We systematically explore the data, preprocess it for optimal model performance, and compare a suite of machine learning algorithms to find the most effective predictor of income level. The project culminates in fine-tuning the champion model to achieve the best possible results.

The project is broken down into three main phases:

1.  **📊 Exploratory Data Analysis (EDA):** Uncovering patterns, correlations, and anomalies in the data.
2.  **🧹 Data Preprocessing:** Cleaning, encoding, scaling, and balancing the dataset to prepare it for modeling.
3.  **🤖 Modeling & Evaluation:** Training, comparing, and tuning multiple classification models.

-----

## 📊 Exploratory Data Analysis (EDA)

Our initial analysis revealed several key insights that guided the feature engineering and modeling process:

  * **Influential Features:** Features like **age, education level, hours-per-week, and marital status** showed a significant relationship with income.
  * **Class Imbalance:** The dataset is imbalanced, with significantly more individuals earning `$<=50K` than `>$50K`. This was a critical consideration for model evaluation and training.
  * **Data Skewness:** Continuous variables like `capital-gain` were highly skewed, requiring normalization.

-----

## 🧹 Data Preprocessing

To prepare the data for our models, we implemented a robust preprocessing pipeline:

  * **Missing Values:** Handled missing values in `work-class`, `occupation`, and `native-country` by replacing them with a distinct 'Unknown' category.
  * **Categorical Encoding:** Used a combination of **One-Hot Encoding** for most categorical features and **Frequency Encoding** for the high-cardinality `native-country` feature.
  * **Feature Scaling:** Applied `RobustScaler` to numerical features to handle outliers and standardize scales.
  * **Balancing Classes:** Utilized the **SMOTE (Synthetic Minority Over-sampling Technique)** to address the class imbalance by generating synthetic samples for the minority class (income \>$50K).

-----

## 🤖 Model Training and Evaluation

We trained and evaluated a diverse set of classification algorithms to identify the top performer.

### Models Tested

  * K-Nearest Neighbors (KNN)
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * MLP (Neural Network)
  * **XGBoost** (Extreme Gradient Boosting)
  * **LightGBM**

### Initial Results (Without Tuning)

The initial comparison showed that ensemble methods, particularly gradient boosting models, were the most promising.

| Model | Accuracy | Precision | Recall | F1-score |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest | 0.84 | 0.66 | 0.70 | 0.68 |
| **LightGBM** | **0.86** | **0.67** | **0.76** | **0.71** |
| **XGBoost** | **0.86** | **0.68** | **0.75** | **0.71** |
| MLP | 0.80 | 0.54 | 0.85 | 0.66 |
| KNN | 0.82 | 0.59 | 0.79 | 0.68 |

-----

## 📈 Final Results & Hyperparameter Tuning

After identifying **XGBoost** as the top-performing model, we used `GridSearch`, `RandomizedSearch`, and `Bayesian Optimization` to fine-tune its hyperparameters. The final model was then validated using 5-fold cross-validation, yielding robust and reliable performance metrics.

### Tuned XGBoost Performance (5-Fold Cross-Validation)

  * **Accuracy:** $0.8971 \pm 0.0026$
  * **AUC:** $0.8971 \pm 0.0026$
  * **Precision:** $0.8848 \pm 0.0029$
  * **Recall:** $0.9132 \pm 0.0036$
  * **F1-Score:** $0.8988 \pm 0.0026$

Statistical comparison using **McNemar's test** confirmed that the performance differences between the models were significant, solidifying XGBoost's superiority for this task.

-----

## 🚀 How to Run

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/income-prediction.git
    cd income-prediction
    ```

2.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis:**
    Open and run the `income_prediction_analysis.ipynb` notebook to see the full workflow from EDA to final model evaluation.

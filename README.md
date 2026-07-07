# JCDSJKTPM-34_Remedial
# 🏦 Bank Marketing Campaign - Term Deposit Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)

A machine learning project predicting term deposit subscriptions for bank marketing campaigns using advanced classification algorithms and imbalanced learning techniques.

## Author
Tengku Arika Hazera


## External Links:

[Streamlit Link](https://jcdsjktpm34final-projectremedial-dfyw9ppnufrp9uz5tqemht.streamlit.app/)

[Tableau Public Link](https://public.tableau.com/views/Final_Project_Remedial/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## Overview

This project is part of the **Purwadhika Data Science Bootcamp Final Capstone Project**. It aims to predict whether a bank client will subscribe to a term deposit based on demographic information, financial status, and previous campaign interactions.

### Key Objectives

- Maximize **Recall** (≥60%) to identify most potential subscribers
- Achieve **PR-AUC** (≥0.50) for reliable predictions on imbalanced data
- Achieve high ROC-AUC score (≥0.80)to distinguish subscribers from non subscribers
- Optimize marketing campaign ROI through targeted client prioritization

## Business Problem

Banks invest significant resources in marketing campaigns for term deposits, but face several challenges:

1. Wasted resources on customers unlikely to subscribe
2. Inefficient allocation of call center time
3. Potential customer fatigue from excessive contact attempts
4. Suboptimal return on investment (ROI) from marketing campaigns

### Solution

`Primary objective`: Generating ML model to analyze customers' data features and build a predictive model to identify customers most likely to subscribe to term deposits, enabling the bank to:

- Optimize marketing campaign efficiency
- Improve customer experience by reducing unnecessary contacts
- Increase overall campaign success rate

`Success metrics`:

- Maximize Recall (minimize missed potential subscribers)
- Achieve high ROC-AUC score
- Achieve strong PR-AUC score for imbalanced data

## Stakeholders
Head of Marketing: Use predictions to prioritize customer contacts


## Dataset

**Source**: [Bank Marketing Campaigns Dataset - Kaggle](https://www.kaggle.com/datasets/volodymyrgavrysh/bank-marketing-campaigns-dataset)

### Dataset Statistics
- **Total Records**: 41,188
- **Features**: 20
- **Target**: Binary (yes/no subscription)
- **Class Distribution**: Imbalanced (~10% positive class)

### Feature Categories

#### 1. Client Demographics
- Age, Job type, Marital status, Education level

#### 2. Financial Status
- Housing loan, Personal loan, Is default status known

#### 3. Campaign Contact Details
- Contact type, Month, Day of week

#### 4. Campaign Statistics
- Number of contacts, Days since last contact, Previous contacts, Previous outcome

#### 5. Macroeconomic Indicators
- Number of employment, consumer confidence index, 3 month EURIBOR Interest rate.

##  Methodology

### 1. Data Understanding & Cleaning
- Missing value analysis and imputation strategy
- Drop duplicates
- Outlier detection and treatment
- Feature type identification and conversion
- Data quality assessment

### 2. Exploratory Data Analysis
- Univariate and bivariate analysis
- Inferential statistics (Chi-square tests)
- Correlation analysis
- Target variable imbalance detection

### 3. Feature Engineering
- Transform `pdays` to binary `was_contacted_before`
- Transform `default` to binary `is default status known`
- Remove unimportant categorical features
- Create preprocessing pipelines and Transformer
- Use Simple Imputer for better imputing result
- Use One-Hot-Encoder for categorical encoding
- Use Ordinal-Encoder for ordinal encoding
- Use Target-Encoder for target encoding

### 4. Model Development

#### Models Benchmarked
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- LightGBM
- XGBoost

#### Handling Class Imbalance
- Class Weight Balancing (For Available Model)

#### Optimization
- Hyperparameter tuning with GridSearchCV
- 5-fold cross-validation

### 5. Model Evaluation
- Confusion Matrix analysis
- ROC-AUC and PR-AUC curves
- Cost-benefit analysis
- SHAP interpretability

## Results

### Model Performance
- **Recall**: Achieved target (66%)
- **ROC-AUC**: Achieved target (0.81)
- **PR-AUC**: Slighly lower compared target target (0.49 out of 0.50)

### Business Impact
   - Estimated net benefit: almost 4.2 times profit increase or `320% more profit` gain compared to baseline approach
   - Enables targeted marketing campaigns with higher efficiency
   - Reduces unnecessary customer contacts improving satisfaction
   - Provides data-driven prioritization for call center operations

### Top Predictive Features
1. Macroeconomic Indicator: nr.employed, consumer confidence index and euribor 3m affects the decision and psychological factors of a person to take a term deposits
2. Age: It shows that younger adults and retired customer are more likely to subscribe
3. Contact type: customer that called using telephone will likely not to subscribe

### Dependencies

- Python 3.8 or higher
- Miniconda
- Virtual environment / Miniconda (recommended)

### Features of Streamlit App
- **Home**: Homepage navigation into other section
- **Single Client Prediction**: Input client details for instant prediction
- **Batch Prediction**: Upload CSV for bulk predictions
- **Model Training**: Section to train the model
- **Model Documentation**: Learn about methodology and results
- **Data Dictionary**: Learn more about the dataset and what it means
- **Macroeconomic Analysis**: Learn about Macroeconomic behind the scene

##  Project Structure

```
bank-marketing-prediction/
│
├── bank_marketing_analysis_amended (1).ipynb     # Complete analysis pipeline
├── app_white (1).py                            # Streamlit web application
├── README.md                         # This file
├── requirements (1).txt                  # Streamlit Requirement file 
├── raw_data.csv                      # Bank Marketing Campaingn unprocessed dataset
├── finalised_model.sav       # Final Model (Tuned LGBM)
└── Final_Project_Remedial (1).twbx                # Tableau files
```

## Model Performance

### Classification Metrics
| Metric | Score | 
|--------|-------|
| **Recall** | 0.66 | 
| ROC-AUC | 0.81 |
| **PR-AUC** | 0.49 |


### Confusion Matrix Interpretation
- **True Positives (TP)**: Correctly identified subscribers
- **True Negatives (TN)**: Correctly identified non-subscribers
- **False Positives (FP)**: Incorrectly predicted subscriptions (wasted calls)
- **False Negatives (FN)**: Missed potential subscribers (lost revenue)

### Cost-Benefit Analysis
- **Cost per contact**: $5 (phone call + agent time)
- **Benefit per subscription**: $50 (term deposit profit)
- **Optimization**: Model minimizes FN (missed revenue) while managing FP (wasted costs)

## Technologies Used

### Core Libraries
- **Python 3.8+**: Programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Imbalanced-learn**: Handling class imbalance

### Visualization
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts for Streamlit app

### Machine Learning Models
- **Ensemble Method**: Random Forest
- **Boosting Frameworks**: XGBoost, LightGBM
- **Linear Model**: Logistic Regression
- **Non Linear Model**: Decision Tree
- **Instance-based**: K-Nearest Neighbors

### Model Interpretation
- **SHAP**: Model explainability and feature importance

### Deployment
- **Streamlit**: Web application framework
- **Joblib**: Model serialization

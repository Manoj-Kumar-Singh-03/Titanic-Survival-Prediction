# Titanic-Survival-Prediction

## Overview
This project implements a machine learning model to predict passenger survival outcomes from the Titanic disaster using the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic). The solution includes comprehensive data preprocessing, feature engineering, and a Random Forest classification model.

## Repository Structure

titanic-survival-prediction/
├── data/
│   ├── tested.xls                # Input dataset
│   └── titanic_survivors.csv     # Preprocessed data (optional)
├── images/
│   └── confusion_matrix.png      # Visualization output
├── survival.ipynb                # Jupyter notebook (optional)
├── survival.py                   # Main training script
└── README.md                     # This documentation


## Requirements
```bash
Python 3.8+ 
pip install -r requirements.txt

requirement.txt
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2


## How to Run

Clone repository:
git clone https://github.com/Manoj-Kumar-Singh-03/Titanic-Survival-Prediction.git

Execute :
python survival.py

Expected Output:
Dataset Info:
RangeIndex: 418 entries, 0 to 417
Data columns (total 12 columns):
...

Preprocessing Completed:
Age              0
Fare             0
Embarked         0
...

Model Performance:
Accuracy: 
Precision: 
Recall: 

Key Features:
Data Cleaning Pipeline: Systematic handling of missing values
Feature Engineering: Combined one-hot encoding with smart feature selection
Reproducibility: Fixed random seeds (42) for consistent results
Model Interpretability: Built-in feature importance visualization

License
MIT License - See LICENSE for details

References
Original Titanic Competition: Kaggle
Scikit-learn Documentation: RandomForestClassifier



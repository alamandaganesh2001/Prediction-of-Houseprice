# House Price Prediction and EDA
# Overview of the project:
This repository contains a house price prediction project with exploratory data analysis (EDA). The goal of the project is to predict the price of houses using regression models.We all have experienced a time when we have to look up for a new house to buy. But then the journey begins with a lot of frauds, negotiating deals, researching the local areas and so on.
House Price Prediction using Machine Learning So to deal with this kind of issues Today we will be preparing a MACHINE LEARNING Based model, trained on the House Price Prediction Dataset.

# Data: 

The dataset used in this project is from the Kaggle 

# Exploratory Data Analysis
The exploratory data analysis notebook is named EDA.ipynb. In this notebook, you will find the following:

# Dataset contains:

1 Id To count the records.

2 MSSubClass Identifies the type of dwelling involved in the sale.

3 MSZoning Identifies the general zoning classification of the sale.

4 LotArea Lot size in square feet.

5 LotConfig Configuration of the lot

6 BldgType Type of dwelling

7 OverallCond Rates the overall condition of the house

8 YearBuilt Original construction year

9 YearRemodAdd Remodel date (same as construction date if no remodeling or additions).

10 Exterior1st Exterior covering on house

11 BsmtFinSF2 Type 2 finished square feet.

12 TotalBsmtSF Total square feet of basement area

13 SalePrice To be predicted


# House Price Prediction
The house price prediction notebook is named House Price Prediction & EDA.ipynb. In this notebook, you will find the following:
# Data loading and basic information about the data.

Feature selection: selecting the most important features based on correlation analysis and feature importance.
Data preprocessing: handling missing data, transforming numerical and categorical variables, and handling outliers.
Model selection: testing different regression models, such as Linear Regression.
Model evaluation: evaluating the performance of each model using mean absolute error.
Model tuning: selecting the best hyperparameters for the best-performing model.
# Requirements
```python
To run the notebooks, you need to have Python 3 installed, along with the following libraries:

Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
You can install these libraries by running pip install -r requirements.txt in your terminal.
```

# Conclusion:

This project shows how to use exploratory data analysis and regression models to predict the sale price of houses. It also demonstrates how to handle missing data, transform numerical and categorical variables. The best-performing model was selected based on cross-validation and different metrics.

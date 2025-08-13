# California Housing Price Prediction with Linear Regression

## Introduction
This project demonstrates how to use supervised machine learning to predict housing prices in California using the Linear Regression algorithm. The notebook walks through the process from data loading and exploration to model building and evaluation.

## Dataset Source
The dataset used is the [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from scikit-learn. It contains information collected from the 1990 California census.

### Feature Names and Descriptions
| Feature        | Description |
|---------------|-------------|
| MedInc        | Median income in block group |
| HouseAge      | Median house age in block group |
| AveRooms      | Average number of rooms per household |
| AveBedrms     | Average number of bedrooms per household |
| Population    | Block group population |
| AveOccup      | Average number of household members |
| Latitude      | Block group latitude |
| Longitude     | Block group longitude |
| MedHouseVal   | Median house value (target variable) |

## Problem Statement
The goal is to predict the median house value (`MedHouseVal`) for California districts using features such as median income, population, and average number of rooms. The dataset is sourced from `sklearn.datasets.fetch_california_housing`.

## Exploratory Data Analysis
- Correlation heatmap to visualize relationships between features.
- Scatter plots with regression lines to show how each feature relates to the target variable.
- Key finding: Median income (`MedInc`) is the most influential feature for predicting house value.

## Data Preprocessing
- Loaded the dataset as a pandas DataFrame.
- Checked for missing values (none found).
- Defined `target` and `features`.
- Split data into training and testing sets (80/20 split).

## Model Building
- Used scikit-learn's `LinearRegression` model.
- Trained the model on the training set.
- Made predictions on the test set.

## Evaluation
- Calculated Mean Squared Error (MSE) to assess model performance.
- Achieved an MSE of approximately 0.5559, meaning the average squared error is about $55,590.

## Conclusion
- Successfully built and evaluated a linear regression model for California housing prices.
- The model provides a baseline for future improvements.
- Further steps could include feature engineering, trying advanced models (Decision Tree, Random Forest, Gradient Boosting), and hyperparameter tuning.

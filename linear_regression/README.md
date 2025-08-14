# California Housing Price Prediction with Linear Regression

## Introduction
This folder demonstrates supervised machine learning for predicting California housing prices using regression algorithms. The notebooks in this folder walk through data loading, exploration, model building, and evaluation using different regression approaches. 

## Notebooks and Their Purpose

- **01_linear_regression.ipynb**: Builds and evaluates a Linear Regression model to predict median house value (`MedHouseVal`). Serves as a baseline for regression performance.
- **02_decision_tree.ipynb**: Builds and evaluates a Decision Tree Regressor on the same dataset. Compares its performance to Linear Regression and explores non-linear relationships.

## Dataset Source
The dataset used is the [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from scikit-learn. It contains information collected from the 1990 California census and is used in both notebooks.

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
Predict the median house value (`MedHouseVal`) for California districts using features such as median income, population, and average number of rooms. The same dataset and features are used in both notebooks for consistent comparison.

## Exploratory Data Analysis (EDA)
Both notebooks perform EDA, including:
- Checking for missing values (none found).
- Visualizing feature distributions.
- Correlation heatmap to visualize relationships between features.

**Key EDA Findings:**
- Median income (`MedInc`) shows the strongest positive linear relationship with house value.
- Average rooms (`AveRooms`) has a positive but weaker relationship.
- House age (`HouseAge`) and population show weak or non-linear relationships.
- Latitude and longitude have complex, non-linear relationships with house value, suggesting location is important but not captured by a simple linear model.
- No missing values in the dataset.

## Model Building

- **Linear Regression (01_linear_regression.ipynb):**
	- Used scikit-learn's `LinearRegression` model.
	- Trained on 80% of the data, tested on 20%.
	- Provided a baseline for regression performance.

- **Decision Tree Regressor (02_decision_tree.ipynb):**
	- Used scikit-learn's `DecisionTreeRegressor`.
	- Trained and tested on the same splits as Linear Regression.
	- Able to capture non-linear relationships in the data.

## Evaluation

- **Linear Regression:**
	- Calculated Mean Squared Error (MSE) and R² score.
	- Achieved an MSE of approximately 0.5559 (average squared error about $55,590).
	- R² score indicates how well the model explains variance in house values.

- **Decision Tree Regressor:**
	- Calculated R² score and RMSE.
	- Decision Tree may outperform Linear Regression if non-linear relationships are present, but can be prone to overfitting.
	- Compared both models using R² and RMSE to determine which fits the data better.

## Conclusion

- Both models were successfully built and evaluated for predicting California housing prices.
- Linear Regression provides a simple, interpretable baseline and works well when relationships are mostly linear.
- Decision Tree Regressor can capture more complex, non-linear patterns, but may overfit if not tuned.
- In this dataset, `MedInc` is the most influential feature for both models.
- Comparing R² and RMSE values helps determine which model is preferable for this task.
- Further improvements could include feature engineering, hyperparameter tuning, and trying ensemble methods (Random Forest, Gradient Boosting).

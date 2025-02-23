import pandas as pd
import numpy as np
import pickle
from sklearn import tree, linear_model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error

# Read and preprocess data
def read_data(file_name):
    df = pd.read_csv(file_name)

    # Convert 'LapTime' column to numeric (removing 'days' and 'seconds' part)
    df['LapTime'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()

    # Encoding categorical columns
    label_encoder = LabelEncoder()
    df['Compound'] = label_encoder.fit_transform(df['Compound'])
    df['Compound2'] = label_encoder.fit_transform(df['Compound2'])

    # Drop non-predictive columns and set target variable
    X = df.drop(['Race', 'Driver', 'Position', 'Rainfall', 'RainFall2'], axis=1)
    y = df['Position']
    return X, y

# Model initialization functions
def decision_tree_regressor():
    return DecisionTreeRegressor(random_state=42)

def decision_tree_classifier(X_train, y_train):
    dtree = DecisionTreeClassifier()
    return dtree.fit(X_train, y_train)

def linear_regression(X_train, y_train):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    return regr

# Hyperparameter tuning
def hyperparameter_tuning(model, xtrain, ytrain, param_grid):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(xtrain, ytrain)
    return grid_search.best_estimator_

# Define hyperparameter tuning grid
def hyperparameter_tuning_param_grid():
    return {
        'max_depth': [1, 2, 3, 4, 5, None],
        'min_samples_split': [2, 3, 5, 6],
        'min_samples_leaf': [1, 2, 4, 5],
        'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1, 0.2],
    }

# Prediction function
def evaluate_model(model, xtest, ytest):
    y_pred = model.predict(xtest)
    mse = mean_squared_error(ytest, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(ytest, y_pred)
    r2 = r2_score(ytest, y_pred)
    msle = mean_squared_log_error(ytest, y_pred)
    medae = median_absolute_error(ytest, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    print(f"Test R²: {r2}")
    print(f"Test MSLE: {msle}")
    print(f"Test Median Absolute Error: {medae}")
    return y_pred

# Train and save the model
if __name__ == "__main__":
    X, y = read_data("data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = decision_tree_regressor()
    param_grid = hyperparameter_tuning_param_grid()
    tuned_model = hyperparameter_tuning(model, X_train, y_train, param_grid)

    # Evaluate
    evaluate_model(tuned_model, X_test, y_test)

    # Save model
    with open("tuned_model.pkl", "wb") as file:
        pickle.dump(tuned_model, file)

    print("Model training complete and saved as tuned_model.pkl")

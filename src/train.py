import pandas as pd
import numpy as np

import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

#Get the processed dataset
df = pd.read_csv("data/processed/nba_processed.csv")
print(df.shape)

# Features and Target
X = df.drop(columns=[
    "PTS",
    "GmSc",
    "Result",
    "+/-"
])
print(X.head(1))

y = df["PTS"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# Split dataset
print(X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed successfully!")


# -------------------------------
# MLflow Experiment
# -------------------------------
mlflow.set_experiment("NBA Player Performance Prediction")


def train_and_log_model(model, model_name):

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n==============================")
        print(model_name)
        print("==============================")
        print("MAE :", mae)
        print("MSE :", mse)
        print("RMSE:", rmse)
        print("R2  :", r2)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(
            model,
            model_name    
            )
        return r2

linear_model = LinearRegression()

decision_tree = DecisionTreeRegressor(
    random_state=42
)

random_forest = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    random_state=42
)

linear_r2 = train_and_log_model(
    linear_model,
    "Linear Regression"
)

tree_r2 = train_and_log_model(
    decision_tree,
    "Decision Tree"
)

forest_r2 = train_and_log_model(
    random_forest,
    "Random Forest"
)

scores = {
    "Decision Tree": tree_r2,
    "Random Forest": forest_r2
}
best_model_name = max(scores, key=scores.get)


print("\n==============================")
print("Best Model:", best_model_name)
print("==============================")

import os

os.makedirs("models", exist_ok=True)

if best_model_name == "Linear Regression":
    best_model = linear_model

elif best_model_name == "Decision Tree":
    best_model = decision_tree

else:
    best_model = random_forest

with mlflow.start_run(run_name="Best Model Registration"):

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        registered_model_name="NBA_Player_Performance_Model"
    )

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Best model saved successfully!")
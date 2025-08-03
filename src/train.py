# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("data/raw/california_housing.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow tracking
mlflow.set_experiment("California_Housing_Regression")

def train_and_log_model(model, model_name, params: dict):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mlflow.log_metric("mse", mse)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} - MSE: {mse}")
        return mse, mlflow.active_run().info.run_id

# Train Linear Regression
lr = LinearRegression()
lr_mse, lr_run_id = train_and_log_model(lr, "LinearRegression", {"fit_intercept": True})

# Train Decision Tree
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_mse, dt_run_id = train_and_log_model(dt, "DecisionTree", {"max_depth": 5})

# Register best model
best_model = "DecisionTree" if dt_mse < lr_mse else "LinearRegression"
best_run_id = dt_run_id if dt_mse < lr_mse else lr_run_id

model_uri = f"runs:/{best_run_id}/{best_model}"
mlflow.register_model(model_uri=model_uri, name="CaliforniaHousingBestModel")
print(f"✅ Registered best model: {best_model}")
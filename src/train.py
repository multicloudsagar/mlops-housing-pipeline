import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("target")], axis=1)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train two models
lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth=5, random_state=42)

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_mse = mean_squared_error(y_test, lr.predict(X_test))
dt_mse = mean_squared_error(y_test, dt.predict(X_test))

# Choose the best model
best_model = dt if dt_mse < lr_mse else lr
print(f"✅ Best Model: {'DecisionTree' if best_model == dt else 'LinearRegression'}")

# Save the best model locally using MLflow
mlflow.sklearn.save_model(sk_model=best_model, path="models/best_model")
print("✅ Model saved to models/best_model/")
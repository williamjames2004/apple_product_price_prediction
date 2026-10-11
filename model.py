import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("data.csv")

# Remove unnecessary columns
data = data.drop(columns=[
    "Date", "Model_Name",
    "Launch_Price_USD", "Current_Price_USD"
], errors="ignore")

# Fill missing values
data = data.ffill().bfill()

# Input and output
X = data.drop("Current_Price_INR", axis=1)
y = pd.to_numeric(data["Current_Price_INR"], errors="coerce")

# Remove rows with missing target
valid_rows = y.notna()
X = X[valid_rows]
y = y[valid_rows]

# Identify column types
categorical = X.select_dtypes(include=["object"]).columns
numerical = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", SimpleImputer(strategy="median"), numerical)
])

# Decision Tree model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    ))
])

# Evaluate when enough records are available
if len(X) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Mean Absolute Error:", round(
        mean_absolute_error(y_test, predictions), 2
    ))

    if y_test.nunique() > 1:
        print("R2 Score:", round(r2_score(y_test, predictions), 4))
    else:
        print("Small dataset: evaluation skipped.")

# Train on all records and save
model.fit(X, y)
joblib.dump(model, "product_price_model.pkl")

print("Product price model trained and saved successfully!")
print("Training records:", len(X))
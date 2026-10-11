
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "product_price_model.pkl")

FEATURES = [
    "Platform",
    "Product_Category",
    "Condition",
    "Launch_Price_INR",
    "Discount_Pct",
    "Sale_Event",
    "Stock_Status",
    "Rating",
    "Reviews_Count"
]

CATEGORICAL = [
    "Platform",
    "Product_Category",
    "Condition",
    "Sale_Event",
    "Stock_Status"
]

NUMERICAL = [
    "Launch_Price_INR",
    "Discount_Pct",
    "Rating",
    "Reviews_Count"
]

model = None
accuracy_info = {
    "mae": None,
    "r2": None,
    "records": 0,
    "evaluation_records": 0
}


def load_and_train_model():
    global model, accuracy_info

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("data.csv was not found in the project folder.")

    data = pd.read_csv(DATA_PATH)

    if "Current_Price_INR" not in data.columns:
        raise ValueError("The dataset must contain Current_Price_INR.")

    required_columns = FEATURES + ["Current_Price_INR"]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: " + ", ".join(missing_columns)
        )

    X = data[FEATURES].copy()
    y = pd.to_numeric(data["Current_Price_INR"], errors="coerce")

    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(float)

    if len(X) == 0:
        raise ValueError("The dataset contains no valid target prices.")

    # Normalize missing-value handling within the model pipeline.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipeline, CATEGORICAL),
        ("num", numerical_pipeline, NUMERICAL)
    ])

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            max_depth=8,
            random_state=42
        ))
    ])

    mae = None
    r2 = None
    evaluation_records = 0

    # Evaluate on a held-out test split when enough records exist.
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        evaluation_model = joblib.clone(model_pipeline) if hasattr(
            joblib, "clone"
        ) else None

        # Use sklearn.clone to create an independent evaluation model.
        from sklearn.base import clone
        evaluation_model = clone(model_pipeline)
        evaluation_model.fit(X_train, y_train)

        predictions = evaluation_model.predict(X_test)
        mae = round(mean_absolute_error(y_test, predictions), 2)
        evaluation_records = len(y_test)

        if len(y_test) >= 2:
            r2 = round(r2_score(y_test, predictions), 4)

    # Fit the final model using all available records.
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, MODEL_PATH)

    model = model_pipeline
    accuracy_info = {
        "mae": mae,
        "r2": r2,
        "records": len(X),
        "evaluation_records": evaluation_records
    }


def get_model():
    global model

    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            load_and_train_model()

    return model


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}
    submitted_values = None

    try:
        current_model = get_model()
    except Exception as exc:
        current_model = None
        error = str(exc)

    if request.method == "POST":
        form_data = {
            "Platform": request.form.get("Platform", "").strip(),
            "Product_Category": request.form.get(
                "Product_Category", ""
            ).strip(),
            "Condition": request.form.get("Condition", "").strip(),
            "Launch_Price_INR": request.form.get(
                "Launch_Price_INR", ""
            ).strip(),
            "Discount_Pct": request.form.get("Discount_Pct", "").strip(),
            "Sale_Event": request.form.get("Sale_Event", "").strip(),
            "Stock_Status": request.form.get("Stock_Status", "").strip(),
            "Rating": request.form.get("Rating", "").strip(),
            "Reviews_Count": request.form.get("Reviews_Count", "").strip()
        }

        try:
            required_fields = [
                "Platform", "Product_Category", "Condition",
                "Launch_Price_INR", "Discount_Pct", "Sale_Event",
                "Stock_Status", "Rating", "Reviews_Count"
            ]

            for field in required_fields:
                if not form_data[field]:
                    raise ValueError(
                        "Please complete every field before predicting."
                    )

            launch_price = float(form_data["Launch_Price_INR"])
            discount = float(form_data["Discount_Pct"])
            rating = float(form_data["Rating"])
            reviews = int(form_data["Reviews_Count"])

            if launch_price < 0:
                raise ValueError("Launch price cannot be negative.")

            if not 0 <= rating <= 5:
                raise ValueError("Rating must be between 0 and 5.")

            if reviews < 0:
                raise ValueError("Reviews count cannot be negative.")

            if not -100 <= discount <= 100:
                raise ValueError(
                    "Discount percentage must be between -100 and 100."
                )

            if current_model is None:
                current_model = get_model()

            submitted_values = pd.DataFrame([{
                "Platform": form_data["Platform"],
                "Product_Category": form_data["Product_Category"],
                "Condition": form_data["Condition"],
                "Launch_Price_INR": launch_price,
                "Discount_Pct": discount,
                "Sale_Event": form_data["Sale_Event"],
                "Stock_Status": form_data["Stock_Status"],
                "Rating": rating,
                "Reviews_Count": reviews
            }], columns=FEATURES)

            prediction = round(
                float(current_model.predict(submitted_values)[0]), 2
            )

        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc)
        except Exception as exc:
            error = "Prediction failed: " + str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        accuracy=accuracy_info,
        project_title="Product Price Prediction",
        current_year=2026
    )


if __name__ == "__main__":
    # Train and evaluate at startup, then save the model.
    try:
        load_and_train_model()
        print("Product price model trained successfully.")
        print("Training records:", accuracy_info["records"])
        print("Mean Absolute Error:", accuracy_info["mae"])
        print("R2 Score:", accuracy_info["r2"])
    except Exception as exc:
        print("Model initialization error:", exc)

    app.run(debug=True)
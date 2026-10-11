import pandas as pd
import joblib

model = joblib.load("product_price_model.pkl")

product = pd.DataFrame([{
    "Platform": input("Platform (Amazon/Flipkart): "),
    "Product_Category": input("Product Category: "),
    "Condition": input("Condition (New/Used): "),
    "Launch_Price_INR": float(input("Launch Price (INR): ")),
    "Discount_Pct": float(input("Discount Percentage: ")),
    "Sale_Event": input("Sale Event (None/Diwali/Other): "),
    "Stock_Status": input("Stock Status (In Stock/Out of Stock): "),
    "Rating": float(input("Rating: ")),
    "Reviews_Count": int(input("Reviews Count: "))
}])

prediction = model.predict(product)[0]

print("\nPredicted Current Price (INR): ₹", round(prediction, 2))
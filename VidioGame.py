import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("video games sales.csv") 
features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
target = 'Global_Sales'

df_clean = df[features + [target]].dropna()

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
intercept = model.intercept_
coefficients = dict(zip(features, model.coef_))

print("=== Regression Analysis Results ===")
print(f"Intercept: {intercept}")
print("Coefficients:")
for feature, coef in coefficients.items():
    print(f"  {feature}: {coef}")
print(f"\nR-squared (R²): {r2}")
print(f"Mean Squared Error (MSE): {mse}")

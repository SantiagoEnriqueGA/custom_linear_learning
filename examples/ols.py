
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linearModels import OrdinaryLeastSquares
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = OrdinaryLeastSquares(fit_intercept=True)
reg.fit(X, y)

r2 = round(r2_score(y, reg.predict(X)), 2)
coef = [round(c, 2) for c in reg.coef_]
intercept = round(reg.intercept_, 2)
formula = reg.get_formula()

print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")

# Output:
# R^2 Score: 0.86
# Regression Coefficients: [27.2, 45.75, 16.29, 24.21, 19.93]
# Regression Intercept: -1.54
# Regression Formula: y = -1.54 + 27.2017 * x_0 + 45.7470 * x_1 + 16.2942 * x_2 + 24.2104 * x_3 + 19.9321 * x_4
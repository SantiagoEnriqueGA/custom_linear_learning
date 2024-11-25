
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linearModels import Ridge
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = Ridge(alpha=1.0, fit_intercept=True)
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
# R^2 Score: 0.65
# Regression Coefficients: [13.3, 23.05, 8.73, 12.14, 9.68]
# Regression Intercept: -0.53
# Regression Formula: y = -0.53 + 13.3011 * x_0 + 23.0484 * x_1 + 8.7292 * x_2 + 12.1384 * x_3 + 9.6810 * x_4
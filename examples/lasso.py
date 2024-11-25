
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linearModels import Lasso
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = Lasso(alpha=1.0, fit_intercept=True)
reg.fit(X, y)

print(f"R^2 Score: {r2_score(y, reg.predict(X)):.2f}")
print(f"Regression Coefficients: {[round(coef, 2) for coef in reg.coef_]}")
print(f"Regression Intercept: {reg.intercept_:.2f}")
print(f"Regression Formula: {reg.get_formula()}")

# Output:
# R^2 Score: 0.74
# Regression Coefficients: [20.79, 35.49, 16.32, 19.58, 8.02]
# Regression Intercept: 14.28
# Regression Formula: y = 14.28 + 20.7884 * x_0 + 35.4879 * x_1 + 16.3211 * x_2 + 19.5828 * x_3 + 8.0185 * x_4
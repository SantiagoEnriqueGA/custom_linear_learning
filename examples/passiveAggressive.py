
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linearModels import PassiveAggressiveRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)

reg = PassiveAggressiveRegressor(C=.001, max_iter=1000, tol=1e-4)
reg.fit(X, y, save_steps=True)

r2 = round(r2_score(y, reg.predict(X)), 2)
coef = [round(c, 2) for c in reg.coef_]
intercept = round(reg.intercept_, 2)
formula = reg.get_formula()

print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")

# Output:
#         Iteration: 0
#         Iteration: 1
#         Iteration: 2
#         Iteration: 3
#         Iteration: 4
#         Iteration: 5
#         Iteration: 6
#         Iteration: 7
# R^2 Score: 0.86
# Regression Coefficients: [27.13, 45.13, 16.69, 23.39, 19.77]
# Regression Intercept: -3.01
# Regression Formula: y = 27.1272 * x_0 + 45.1323 * x_1 + 16.6942 * x_2 + 23.3948 * x_3 + 19.7663 * x_4 + -3.01   
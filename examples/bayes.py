
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linearModels import Bayesian
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)

reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = True)
alpha_1, alpha_2, lambda_1, lambda_2 = reg.tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
reg.fit(X, y)
print(f"Best Hyperparameters: alpha_1={alpha_1:.2f}, alpha_2={alpha_2:.2f}, lambda_1={lambda_1:.2f}, lambda_2={lambda_2:.2f}")

print("Results after tuning")
print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")

# Output:
#         Improved after, No of iterations: 0, MSE: 670.82
#         Improved after, No of iterations: 1, MSE: 670.82
#         Improved after, No of iterations: 2, MSE: 670.82
#         ...
#         Improved after, No of iterations: 48, MSE: 670.82
#         Improved after, No of iterations: 49, MSE: 670.82
# Stopped after 1000 iterations.
# Best Hyperparameters: alpha_1=0.50, alpha_2=0.27, lambda_1=0.50, lambda_2=0.27
# R^2 Score: 0.86
# Regression Coefficients: [27.19 45.74 16.29 24.15 19.95]
# Regression Intercept: 27.19
# Regression Formula: y = 27.19 + 27.1907 * x_0 + 45.7403 * x_1 + 16.2937 * x_2 + 24.1582 * x_3 + 19.9573 * x_4
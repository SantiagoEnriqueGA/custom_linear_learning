
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from linearModels import RANSAC
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = RANSAC(n=20, k=300, t=0.01, d=10, model=None, 
                 auto_scale_t=True, scale_t_factor=2,
                 auto_scale_n=False, scale_n_factor=2                
                 )
reg.fit(X, y)

r2 = round(r2_score(y, reg.predict(X)), 2)
coef = [round(c, 2) for c in reg.best_fit.coef_]
intercept = round(reg.best_fit.intercept_, 2)
formula = reg.get_formula()

print(f"R^2 Score: {r2}")
print(f"Regression Coefficients: {coef}")
print(f"Regression Intercept: {intercept}")
print(f"Regression Formula: {formula}")

# Output:
#         No model fit, scaling threshold from 0.01 to 0.02
#         No model fit, scaling threshold from 0.02 to 0.04
#         No model fit, scaling threshold from 0.04 to 0.08
#         No model fit, scaling threshold from 0.08 to 0.16
# R^2 Score: 0.82
# Regression Coefficients: [33.79, 42.26, 13.45, 26.59, 25.77]
# Regression Intercept: 7.74
# Regression Formula: y = 7.74 + 33.7898 * x_0 + 42.2570 * x_1 + 13.4509 * x_2 + 26.5918 * x_3 + 25.7650 * x_4
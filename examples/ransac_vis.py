import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples import ols
from linearModels import RANSAC
from linearModels import OrdinaryLeastSquares

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Example data, contains many outliers 
X = np.array([-0.848,-0.800,-0.704,-0.632,-0.488,-0.472,-0.368,-0.336,-0.280,-0.200,-0.00800,-0.0840,0.0240,0.100,0.124,0.148,0.232,0.236,0.324,0.356,0.368,0.440,0.512,0.548,0.660,0.640,0.712,0.752,0.776,0.880,0.920,0.944,-0.108,-0.168,-0.720,-0.784,-0.224,-0.604,-0.740,-0.0440,0.388,-0.0200,0.752,0.416,-0.0800,-0.348,0.988,0.776,0.680,0.880,-0.816,-0.424,-0.932,0.272,-0.556,-0.568,-0.600,-0.716,-0.796,-0.880,-0.972,-0.916,0.816,0.892,0.956,0.980,0.988,0.992,0.00400]).reshape(-1, 1)
y = np.array([-0.917,-0.833,-0.801,-0.665,-0.605,-0.545,-0.509,-0.433,-0.397,-0.281,-0.205,-0.169,-0.0531,-0.0651,0.0349,0.0829,0.0589,0.175,0.179,0.191,0.259,0.287,0.359,0.395,0.483,0.539,0.543,0.603,0.667,0.679,0.751,0.803,-0.265,-0.341,0.111,-0.113,0.547,0.791,0.551,0.347,0.975,0.943,-0.249,-0.769,-0.625,-0.861,-0.749,-0.945,-0.493,0.163,-0.469,0.0669,0.891,0.623,-0.609,-0.677,-0.721,-0.745,-0.885,-0.897,-0.969,-0.949,0.707,0.783,0.859,0.979,0.811,0.891,-0.137]).reshape(-1)

ransac_reg = RANSAC(n=10, k=1000, t=0.001, d=35, model=None,
            auto_scale_t=True, scale_t_factor=1.5,
             )
ransac_reg.fit(X, y)

ols_reg = OrdinaryLeastSquares()
ols_reg.fit(X, y)

# Visualize the regression line
line = np.linspace(-1, 1, num=100).reshape(-1,1)

plt.figure(figsize=(10, 10))
sns.set_theme(style="whitegrid")

plt.scatter(X, y, label="Data points", color="blue", edgecolor='k')
plt.plot(line, ransac_reg.predict(line), c="peru", label="RANSAC Regression Line", linewidth=2) 
plt.plot(line, ols_reg.predict(line), c="red", label="OLS Regression Line", linewidth=2)

plt.title("RANSAC Regression Visualization")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/ransac_vis.png", dpi=300)


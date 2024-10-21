# Importing the required libraries
from git import Object
from git.repo import Repo
import numpy as np
from math import log, floor, ceil

class Utility(object):
    """
    Utility class
    """
    pass


import numpy as np

class OrdinaryLeastSquares(object):
    """
    Ordinary Least Squares (OLS) class
    """

    def __init__(self, fit_intercept=True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the model to the data
        """
        if self.fit_intercept:                              # If fit_intercept is True
            X = np.hstack([np.ones((X.shape[0], 1)), X])    # Add a column of ones to X, for the intercept
        
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y       # Compute the coefficients using the normal equation, w = (X^T * X)^-1 * X^T * y
        
        if self.fit_intercept:                              # If fit_intercept is True
            self.intercept_ = self.coef_[0]                 # Set the intercept to the first element of the coefficients
            self.coef_ = self.coef_[1:]                     # Set the coefficients to the remaining elements
        
        else:                                               # Else if fit_intercept is False
            self.intercept_ = 0.0                           # Set the intercept to 0.0
                
    def predict(self, X):
        """
        Predict using the linear model
        """
        if self.fit_intercept:                                  # If fit_intercept is True
            X = np.hstack([np.ones((X.shape[0], 1)), X])        # Add a column of ones to X, for the intercept
            return X @ np.hstack([self.intercept_, self.coef_]) # Return the predicted values
        
        else:                                                   # Else if fit_intercept is False
            return X @ self.coef_                               # Return the predicted values
        
    def get_formula(self):
        """
        Returns the formula of the model
        """
        terms = [f"{coef:.4f} * x_{i}" for i, coef in enumerate(self.coef_)]    # Create the terms of the formula
        formula = " + ".join(terms)                                             # Join the terms with " + "
        if self.fit_intercept:                                                  # If fit_intercept is True
            formula = f"{self.intercept_:.2f} + " + formula                     # Add the intercept to the formula
        return f"y = {formula}" 
        
class Ridge(object):
    """
    Ridge Regression Class
    """
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the model to the data using coordinate descent
        """
        if self.fit_intercept:                                  # If fit_intercept is True
            X = np.hstack([np.ones((X.shape[0], 1)), X])        # Add a column of ones to X, for the intercept
        
        n_samples, n_features = X.shape                         # Get the number of samples and features
        self.coef_ = np.zeros(n_features)                       # Initialize the coefficients to zeros
        
        for iteration in range(self.max_iter):                  # For each iteration
            coef_old = self.coef_.copy()                        # Copy the coefficients
            
            for j in range(n_features):                         # For each feature
                residual = y - X @ self.coef_                   # Compute the residuals
                rho = X[:, j] @ residual                        # Compute rho, the correlation between the feature and the residuals
                
                if j == 0 and self.fit_intercept:               # If it's the intercept term
                    self.coef_[j] = rho / (X[:, j] @ X[:, j])   # Update the coefficient
                
                else:                                           # Else, update the coefficient using the Ridge formula
                    self.coef_[j] = rho / (X[:, j] @ X[:, j] + self.alpha)
            
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:    # If the coefficients have converged
                break
        
        if self.fit_intercept:                  # If fit_intercept is True
            self.intercept_ = self.coef_[0]     # Set the intercept to the first element of the coefficients
            self.coef_ = self.coef_[1:]         # Set the coefficients to the remaining elements
        
        else:                                   # Else if fit_intercept is False
            self.intercept_ = 0.0               # Set the intercept to 0.0
    
    def predict(self, X):
        """
        Predict using the linear model
        """
        if self.fit_intercept:                                  # If fit_intercept is True
            X = np.hstack([np.ones((X.shape[0], 1)), X])        # Add a column of ones to X, for the intercept
            return X @ np.hstack([self.intercept_, self.coef_]) # Return the predicted values
        
        else:                                                   # Else if fit_intercept is False
            return X @ self.coef_                               # Return the predicted values
        
    def get_formula(self):
        """
        Returns the formula of the model
        """
        terms = [f"{coef:.4f} * x_{i}" for i, coef in enumerate(self.coef_)]    # Create the terms of the formula
        formula = " + ".join(terms)                                             # Join the terms with " + "
        if self.fit_intercept:                                                  # If fit_intercept is True
            formula = f"{self.intercept_:.2f} + " + formula                     # Add the intercept to the formula
        return f"y = {formula}" 

class Lasso(object):
    """
    Lasso Regression Class
    """
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """
        Fit the model to the data using coordinate descent
        """
        if self.fit_intercept:                              # If fit_intercept is True
            X = np.hstack([np.ones((X.shape[0], 1)), X])    # Add a column of ones to X, for the intercept
        
        n_samples, n_features = X.shape                     # Get the number of samples and features
        self.coef_ = np.zeros(n_features)                   # Initialize the coefficients to zeros
        
        for iteration in range(self.max_iter):              # For each iteration
            coef_old = self.coef_.copy()                    # Copy the coefficients
            
            for j in range(n_features):                     # For each feature
                residual = y - X @ self.coef_               # Compute the residuals
                rho = X[:, j] @ residual                    # Compute rho
                
                if j == 0 and self.fit_intercept:           # If it's the intercept term
                    self.coef_[j] = rho / n_samples         # Update the coefficient
                else:                                       # Else, update the coefficient using the Lasso formula
                    self.coef_[j] = np.sign(rho) * max(0, abs(rho) - self.alpha) / (X[:, j] @ X[:, j])
            
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tol:    # If the coefficients have converged
                break
        
        if self.fit_intercept:                              # If fit_intercept is True
            self.intercept_ = self.coef_[0]                 # Set the intercept to the first element of the coefficients
            self.coef_ = self.coef_[1:]                     # Set the coefficients to the remaining elements
        
        else:                                               # Else if fit_intercept is False
            self.intercept_ = 0.0                           # Set the intercept to 0.0
    
    def predict(self, X):
        """
        Predict using the linear model
        """
        if self.fit_intercept:                                      # If fit_intercept is True
            X = np.hstack([np.ones((X.shape[0], 1)), X])            # Add a column of ones to X, for the intercept
            return X @ np.hstack([self.intercept_, self.coef_])     # Return the predicted values
        
        else:                                                       # Else if fit_intercept is False
            return X @ self.coef_                                   # Return the predicted values
        
    def get_formula(self):
        """
        Returns the formula of the model
        """
        terms = [f"{coef:.4f} * x_{i}" for i, coef in enumerate(self.coef_)]    # Create the terms of the formula
        formula = " + ".join(terms)                                             # Join the terms with " + "
        if self.fit_intercept:                                                  # If fit_intercept is True
            formula = f"{self.intercept_:.2f} + " + formula                     # Add the intercept to the formula
        return f"y = {formula}"


from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=1, noise=20, random_state=42)
to_predict = np.array([[1]])

# Example Usage OLS (Ordinary Least Squares) Regression
# ----------------------------------------------------------------------------
reg = OrdinaryLeastSquares(fit_intercept=True)
reg.fit(X, y)

print("\nExample Usage OLS (Ordinary Least Squares) Regression")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
print(f"Regression Formula: {reg.get_formula()}")


# Example Usage Ridge Regression
# ----------------------------------------------------------------------------
reg = Ridge(alpha=0.5, fit_intercept=True)
reg.fit(X, y)

print("\nExample Usage Ridge Regression")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
print(f"Regression Formula: {reg.get_formula()}")


# Example Usage Lasso Regression
# ----------------------------------------------------------------------------
reg = Lasso(alpha=0.1, fit_intercept=True)
reg.fit(X, y)

print("\nExample Usage Lasso Regression")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
print(f"Regression Formula: {reg.get_formula()}")



import matplotlib.pyplot as plt

# Plotting the points X and y
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='blue', label='Data Points')

# Plotting the OLS regression line
reg = OrdinaryLeastSquares(fit_intercept=True)
reg.fit(X, y)
plt.plot(X, reg.predict(X), color='red', label='OLS Regression Line')

# Plotting the Ridge regression line
reg = Ridge(alpha=0.5, fit_intercept=True)
reg.fit(X, y)
plt.plot(X, reg.predict(X), color='green', label='Ridge Regression Line')

# Plotting the Lasso regression line
reg = Lasso(alpha=0.1, fit_intercept=True)
reg.fit(X, y)
plt.plot(X, reg.predict(X), color='orange', label='Lasso Regression Line')

# Adding labels and legend
plt.xlabel('Feature 0')
plt.ylabel('Target')
plt.title('Regression Lines')
plt.legend()
plt.show()
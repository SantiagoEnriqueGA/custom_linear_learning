# Importing the required libraries
from git import Object
from git.repo import Repo
from networkx import sigma
import numpy as np
from math import log, floor, ceil
from scipy import linalg

class Utility(object):
    """
    Utility class
    """
    pass


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
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=10000, tol=1e-4):
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
        
        for _ in range(self.max_iter):                  # For each iteration
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
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=10000, tol=1e-4):
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
        
        for _ in range(self.max_iter):                      # For each iteration
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

class Bayesian(Object):
    """
    Bayesian Regression Class
    y = 
    
    Args:
        max_iter: int, default=300
            The maximum number of iterations to perform.
        tol: float, default=0.001
            The convergence threshold. The algorithm will stop if the coefficients change less than the threshold.
        alpha_1: float, default=1e-06
            The shape parameter for the prior on the weights.
        alpha_2: float, default=1e-06
            The scale parameter for the prior on the weights.
        lambda_1: float, default=1e-06
            The shape parameter for the prior on the noise.
        lambda_2: float, default=1e-06
            The scale parameter for the prior on the noise.        
    """
    def __init__(self, max_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = None):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept
    
        self.intercept_ = None
        self.coef_ = None
        
    def fit(self, X, y):
        """
        Fit the model to the data.
        """
        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps      # Machine epsilon, the smallest number that can be added to 1.0 to get a larger number

        # alpha_ is the precision of the weights, lambda_ is the precision of the noise
        alpha_ = 1.0 / (np.var(y) + eps)    # Add `eps` in the denominator to omit division by zero if `np.var(y)` is zero  
        lambda_ = 1.0                       # Initialize the noise precision to 1.0
        
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        
        # Variables to store the values of the parameters from the previous iteration
        self.scores_ = list()
        coef_old_ = None
        
        XT_y = np.dot(X.T, y)                           # Compute X^T * y
        U, S, Vh = linalg.svd(X, full_matrices=False)   # Compute the Singular Value Decomposition of X, X = U * S * Vh
        eigen_vals_ = S**2                              # Compute the eigenvalues of X
        
        # Main loop for the algorithm
        for iter in range(self.max_iter):
            # Update the coefficients
            # coef_ formula: coef_ = Vh * (S^2 / (S^2 + lambda_ / alpha_)) * U^T * y
            coef_ = np.linalg.multi_dot([Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis], XT_y])
            
            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum(1 - alpha_ * eigen_vals_) / np.sum(coef_ ** 2)                      # Compute gamma
            lambda_ = (gamma_ + 2 * lambda_1 - 1) / (np.sum(coef_ ** 2) + 2 * lambda_2)         # Update lambda
            alpha_ = (X.shape[0] - gamma_ + 2 * alpha_1 - 1) / (np.sum(y ** 2) + 2 * alpha_2)   # Update alpha
            
            # Check for convergence
            if coef_old_ is not None and np.sum(np.abs(coef_ - coef_old_)) < self.tol:
                print(f"Converged in {iter} iterations.")
                break
            coef_old_ = np.copy(coef_)  # Copy the coefficients
        
        self.n_iter_ = iter + 1 
        self.coef_ = coef_
        self.alpha_ = alpha_
        self.lambda_ = lambda_

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        self.sigma_ = np.linalg.inv(np.dot(X.T, X) + lambda_ / alpha_ * np.eye(X.shape[1]))
        
        if self.fit_intercept:                              # If fit_intercept is True
            self.intercept_ = self.coef_[0]                 # Set the intercept to the first element of the coefficients
        
        return self
    
    def predict(self, X):
        """
        Predict using the linear model
        """
        return np.dot(X, self.coef_)
    
    def get_formula(self):
        """
        Returns the formula of the model
        """
        terms = [f"{coef:.4f} * x_{i}" for i, coef in enumerate(self.coef_)]    # Create the terms of the formula
        formula = " + ".join(terms)                                             # Join the terms with " + "
        if self.fit_intercept:                                                  # If fit_intercept is True
            formula = f"{self.intercept_:.2f} + " + formula                     # Add the intercept to the formula
        return f"y = {formula}"

if __name__ == "__main__":

    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)
    to_predict = np.array([[1,2,3,4,5]])

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

    # Example Usage Lasso Regression
    # ----------------------------------------------------------------------------
    reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = True)
    reg.fit(X, y)

    print("\nExample Usage Bayesian Regression")
    print(f"Regression Coefficients: {reg.coef_}")
    print(f"Regression Intercept: {reg.intercept_}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")
    
    

    # Example plot
    # ----------------------------------------------------------------------------
    import matplotlib.pyplot as plt
    X, y = make_regression(n_samples=1000, n_features=1, noise=15, random_state=42)

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
    plt.plot(X, reg.predict(X), color='green', label='Ridge Regression Line, Alpha=0.5')

    # Plotting the Lasso regression line
    reg = Lasso(alpha=0.5, fit_intercept=True)
    reg.fit(X, y)
    plt.plot(X, reg.predict(X), color='orange', label='Lasso Regression Line, Alpha=0.5')
    
    # Plotting the Bayesian regression line
    reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = True)
    reg.fit(X, y)
    plt.plot(X, reg.predict(X), color='purple', label='Bayesian Regression Line')

    # Adding labels and legend
    plt.xlabel('Feature 0')
    plt.ylabel('Target')
    plt.title('Regression Lines')
    plt.legend()
    plt.show()


    # # Example plot, Ridge regression lines for different alpha values
    # # ----------------------------------------------------------------------------
    # # Plotting the points X and y
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 0], y, color='blue', label='Data Points')

    # # Plotting the Ridge regression lines for different alpha values
    # alphas = [0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    # colors = ['green', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'yellow', 'black', 'lime']

    # for alpha, color in zip(alphas, colors):
    #     reg = Ridge(alpha=alpha, fit_intercept=True)
    #     reg.fit(X, y)
    #     plt.plot(X, reg.predict(X), color=color, label=f'Alpha={alpha}')

    # # Adding labels and legend
    # plt.xlabel('Feature 0')
    # plt.ylabel('Target')
    # plt.title('Ridge Regression Lines Different Alpha Values')
    # plt.legend()
    # plt.show()



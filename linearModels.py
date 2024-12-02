# Importing the required libraries
import re
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

class Bayesian(object):
    """
    Bayesian Regression Class
   
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
                # print(f"Converged in {iter} iterations.")
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
    
    def tune(self, X, y, beta1=0.9, beta2=0.999, iter=1000):
        """
        Automatically tune the hyperparameters alpha_1, alpha_2, lambda_1, lambda_2
        """
       
        # Initialize the best values of the hyperparameters
        best_alpha_1 = None
        best_alpha_2 = None
        best_lambda_1 = None
        best_lambda_2 = None
        
        self.alpha_1 = 1e-02
        self.alpha_2 = 1e-02
        self.lambda_1 = 1e-02
        self.lambda_2 = 1e-02
        
        # Initialize the best MSE
        best_mse = float('inf')
        
        # Use ADAM optimizer to tune the hyperparameters
        beta1 = beta1         # Exponential decay rate for the first moment estimates
        beta2 = beta2       # Exponential decay rate for the second moment estimates
        epsilon = 1e-8      # A small constant to prevent division by zero
        m_alpha_1 = 0       # The first moment for alpha_1
        v_alpha_1 = 0       # The second moment for alpha_1
        m_alpha_2 = 0       # The first moment for alpha_2
        v_alpha_2 = 0       # The second moment for alpha_2
        m_lambda_1 = 0      # The first moment for lambda_1
        v_lambda_1 = 0      # The second moment for lambda_1
        m_lambda_2 = 0      # The first moment for lambda_2
        v_lambda_2 = 0      # The second moment for lambda_2
        t = 0               # The time step
        
        def _compute_gradient_alpha_1():
            """Compute the gradient of the loss function with respect to alpha_1"""
            return -0.5 * (np.sum(self.coef_ ** 2) + 2 * self.alpha_2)

        def _compute_gradient_alpha_2():
            """Compute the gradient of the loss function with respect to alpha_2"""
            return -0.5 * (self.alpha_1 / self.alpha_2 ** 2)

        def _compute_gradient_lambda_1():
            """Compute the gradient of the loss function with respect to lambda_1"""
            return -0.5 * (np.sum(self.coef_ ** 2) + 2 * self.lambda_2)

        def _compute_gradient_lambda_2():
            """Compute the gradient of the loss function with respect to lambda_2"""
            return -0.5 * (self.lambda_1 / self.lambda_2 ** 2)
        
        # loop untill convergence
        while True:
            # Fit the model
            self.fit(X, y)
            
            # Compute the mean squared error
            mse = np.mean((y - self.predict(X)) ** 2)
            if mse < best_mse:
                print(f"\tImproved after, No of iterations: {t}, MSE: {mse:.2f}")
                best_mse = mse
                best_alpha_1 = self.alpha_1
                best_alpha_2 = self.alpha_2
                best_lambda_1 = self.lambda_1
                best_lambda_2 = self.lambda_2
            
            # Compute the gradients
            grad_alpha_1 = _compute_gradient_alpha_1()
            grad_alpha_2 = _compute_gradient_alpha_2()
            grad_lambda_1 =_compute_gradient_lambda_1()
            grad_lambda_2 =_compute_gradient_lambda_2()
            
            # Update the moving averages of the gradients
            m_alpha_1 = beta1 * m_alpha_1 + (1 - beta1) * grad_alpha_1
            v_alpha_1 = beta2 * v_alpha_1 + (1 - beta2) * grad_alpha_1**2
            m_alpha_2 = beta1 * m_alpha_2 + (1 - beta1) * grad_alpha_2
            v_alpha_2 = beta2 * v_alpha_2 + (1 - beta2) * grad_alpha_2**2
            m_lambda_1 = beta1 * m_lambda_1 + (1 - beta1) * grad_lambda_1
            v_lambda_1 = beta2 * v_lambda_1 + (1 - beta2) * grad_lambda_1**2
            m_lambda_2 = beta1 * m_lambda_2 + (1 - beta1) * grad_lambda_2
            v_lambda_2 = beta2 * v_lambda_2 + (1 - beta2) * grad_lambda_2**2
            
            # Compute the bias-corrected estimates
            m_alpha_1_hat = m_alpha_1 / (1 - beta1**(t + 1))
            v_alpha_1_hat = v_alpha_1 / (1 - beta2**(t + 1))
            m_alpha_2_hat = m_alpha_2 / (1 - beta1**(t + 1))
            v_alpha_2_hat = v_alpha_2 / (1 - beta2**(t + 1))
            m_lambda_1_hat = m_lambda_1 / (1 - beta1**(t + 1))
            v_lambda_1_hat = v_lambda_1 / (1 - beta2**(t + 1))
            m_lambda_2_hat = m_lambda_2 / (1 - beta1**(t + 1))
            v_lambda_2_hat = v_lambda_2 / (1 - beta2**(t + 1))
            
            # Update the hyperparameters
            self.alpha_1 -= 0.01 * m_alpha_1_hat / (np.sqrt(v_alpha_1_hat) + epsilon)
            self.alpha_2 -= 0.01 * m_alpha_2_hat / (np.sqrt(v_alpha_2_hat) + epsilon)
            self.lambda_1 -= 0.01 * m_lambda_1_hat / (np.sqrt(v_lambda_1_hat) + epsilon)
            self.lambda_2 -= 0.01 * m_lambda_2_hat / (np.sqrt(v_lambda_2_hat) + epsilon)
            
            # Check for convergence, if the gradients are close to zero
            if np.abs(grad_alpha_1) < 1e-6 and np.abs(grad_alpha_2) < 1e-6 and np.abs(grad_lambda_1) < 1e-6 and np.abs(grad_lambda_2) < 1e-6:
                print(f"Converged in {t} iterations.")
                break
            elif t >= iter:
                print(f"Stopped after {t} iterations.")
                break
            t+=1
        
        # Fit the model with the best hyperparameters
        self.alpha_1 = best_alpha_1
        self.alpha_2 = best_alpha_2
        self.lambda_1 = best_lambda_1
        self.lambda_2 = best_lambda_2
        reg.fit(X, y)
            
        # Return the best hyperparameters, and fitted model
        return best_alpha_1, best_alpha_2, best_lambda_1, best_lambda_2
        
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

class RANSAC(object):
    """
    RANSAC (Random Sample Consensus) class
    """
    def __init__(self, n=10, k=100, t=0.05, d=10, model=None, 
                 auto_scale_t=False, scale_t_factor=2,
                 auto_scale_n=False, scale_n_factor=2
                 ):
        """
        Initialize the linear model with the given parameters.
        """
        # Can only scale one of the threshold or the number of data points
        assert not (auto_scale_t and auto_scale_n), "Can only scale one of the threshold or the number of data points"
        
        # Default model is Ordinary Least Squares
        self.model = OrdinaryLeastSquares(fit_intercept=True) if model is None else model
        
        self.n = n                  # Number of data points to estimate parameters
        self.k = k                  # Maximum iterations allowed
        self.t = t                  # Threshold value to determine if points are fit well, in terms of residuals
        self.d = d                  # Number of close data points required to assert model fits well
                
        self.best_fit = None        # Best model fit
        self.best_error = np.inf    # Best error
        
        # If after all iterations, the model was not fit, scale the threshold until it fits
        self.scale_threshold = auto_scale_t
        self.scale_t_factor = scale_t_factor
        
        # If after all iterations, the model was not fit, scale the number of data points until it fits
        self.scale_n = auto_scale_n
        self.scale_n_factor = scale_n_factor
    
    def _square_loss(self, y_true, y_pred):
        """
        Compute the square loss
        """
        return (y_true - y_pred) ** 2
    
    def _mean_square_loss(self, y_true, y_pred):
        """
        Compute the mean square loss
        """
        return np.mean(self._square_loss(y_true, y_pred))
    
    def fit(self, X, y):
        """
        Fit the model to the data, using RANSAC
        """
        for _ in range(self.k):
            # Randomly select n data points
            idx = np.random.choice(X.shape[0], self.n, replace=False)
            X_subset, y_subset = X[idx], y[idx]
            
            # Fit the model
            self.model.fit(X_subset, y_subset)
            
            # Compute the residuals
            y_pred = self.model.predict(X)
            residuals = np.abs(y - y_pred)
            
            # Compute the inliers, the data points that are fit well
            inliers = residuals < self.t
            
            # Check if the model fits well
            if np.sum(inliers) > self.d:
                # Update the best model
                error = np.sum(self._square_loss(y, y_pred))
                if error < self.best_error:
                    self.best_error = error
                    self.best_fit = self.model
        
        # If the model was not fit, scale the threshold
        if self.best_fit is None and self.scale_threshold:
            print(f"\tNo model fit, scaling threshold from {self.t:.2} to {(self.scale_t_factor * self.t):.2}")
            self.t *= self.scale_t_factor
            self.fit(X, y)
        
        # If the model was not fit, scale the number of data points
        if self.best_fit is None and self.scale_n:
            if (self.scale_n_factor * self.n) > X.shape[0]:
                raise ValueError(f"Cannot scale number of data points beyond the number of data points available.")               
            
            print(f"\tNo model fit, scaling number of data points from {self.n} to {self.scale_n_factor * self.n}")
            self.n *= self.scale_n_factor
            self.fit(X, y)
            
    def predict(self, X):
        """
        Predict using the best fit model
        """
        return self.best_fit.predict(X)
    
    def get_formula(self):
        """
        Returns the formula of the model
        """
        try:
            return self.best_fit.get_formula()
        except:
            return "No model fit available"

if __name__ == "__main__":
    from sklearn.metrics import r2_score

    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
    to_predict = np.array([[1,2,3,4,5]])
    
    # import pandas as pd
    # df = pd.read_csv('data/carsDotCom_prepared.csv')
    # X = df.drop(columns=['Price']).values
    # y = df['Price'].values
    
    to_predict = X[0].reshape(1, -1)

    # Example Usage OLS (Ordinary Least Squares) Regression
    # ----------------------------------------------------------------------------
    reg = OrdinaryLeastSquares(fit_intercept=True)
    reg.fit(X, y)

    print("\nExample Usage OLS (Ordinary Least Squares) Regression")
    print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
    print(f"Regression Coefficients: {reg.coef_}")
    print(f"Regression Intercept: {reg.intercept_}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")


    # Example Usage Ridge Regression
    # ----------------------------------------------------------------------------
    reg = Ridge(alpha=0.5, fit_intercept=True)
    reg.fit(X, y)

    print("\nExample Usage Ridge Regression")
    print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
    print(f"Regression Coefficients: {reg.coef_}")
    print(f"Regression Intercept: {reg.intercept_}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")


    # Example Usage Lasso Regression
    # ----------------------------------------------------------------------------
    reg = Lasso(alpha=0.1, fit_intercept=True)
    reg.fit(X, y)

    print("\nExample Usage Lasso Regression")
    print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
    print(f"Regression Coefficients: {reg.coef_}")
    print(f"Regression Intercept: {reg.intercept_}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")

    # Example Usage Lasso Regression
    # ----------------------------------------------------------------------------
    reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = True)
    reg.fit(X, y)

    print("\nExample Usage Bayesian Regression")
    print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
    print(f"Regression Coefficients: {reg.coef_}")
    print(f"Regression Intercept: {reg.intercept_}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")
    
    print("\nExample Usage Bayesian Regression with Hyperparameter Tuning")
    reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = True)
    alpha_1, alpha_2, lambda_1, lambda_2 = reg.tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
    reg.fit(X, y)
    print(f"Best Hyperparameters: alpha_1={alpha_1:.2f}, alpha_2={alpha_2:.2f}, lambda_1={lambda_1:.2f}, lambda_2={lambda_2:.2f}")
    
    # reg = Bayesian(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, fit_intercept = True)
    print("Results after tuning")
    print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
    print(f"Regression Coefficients: {reg.coef_}")
    print(f"Regression Intercept: {reg.intercept_}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")
    
    # Example Usage RADAR (Random Sample Consensus) Regression
    # ----------------------------------------------------------------------------
    reg = RANSAC(n=20, k=300, t=0.01, d=10, model=None, 
                 auto_scale_t=True, scale_t_factor=2,
                 auto_scale_n=False, scale_n_factor=2                
                 )
    reg.fit(X, y)
    
    print("\nExample Usage RANSAC (Random Sample Consensus) Regression")
    print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
    print(f"Predicted Value for {to_predict}: {reg.predict(to_predict)}")
    print(f"Regression Formula: {reg.get_formula()}")
    
    

    # # # Example plot
    # # ----------------------------------------------------------------------------
    # import matplotlib.pyplot as plt
    
    # X, y = make_regression(n_samples=1000, n_features=1, noise=15, random_state=42)

    # # Plotting the points X and y
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 0], y, color='blue', label='Data Points')

    # # Plotting the OLS regression line
    # reg = OrdinaryLeastSquares(fit_intercept=True)
    # reg.fit(X, y)
    # plt.plot(X, reg.predict(X), color='red', label='OLS Regression Line')

    # # Plotting the Ridge regression line
    # reg = Ridge(alpha=0.5, fit_intercept=True)
    # reg.fit(X, y)
    # plt.plot(X, reg.predict(X), color='green', label='Ridge Regression Line, Alpha=0.5')

    # # Plotting the Lasso regression line
    # reg = Lasso(alpha=0.5, fit_intercept=True)
    # reg.fit(X, y)
    # plt.plot(X, reg.predict(X), color='orange', label='Lasso Regression Line, Alpha=0.5')
    
    # # Plotting the Bayesian regression line
    # reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1, lambda_1=1, lambda_2=1e-06, fit_intercept = True)
    # reg.fit(X, y)
    # plt.plot(X, reg.predict(X), color='purple', label='Bayesian Regression Line')
    
    # # Plotting the Bayesian regression line, after tuning
    # reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept = True)
    # alpha_1, alpha_2, lambda_1, lambda_2 = reg.tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
    # reg.fit(X, y)
    # plt.plot(X, reg.predict(X), color='brown', label='Bayesian Regression Line, Tuned')
    

    # # Adding labels and legend
    # plt.xlabel('Feature 0')
    # plt.ylabel('Target')
    # plt.title('Regression Lines')
    # plt.legend()
    # plt.show()


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



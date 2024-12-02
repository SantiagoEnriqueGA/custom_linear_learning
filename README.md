# Custom Linear Learning

## Goal
This project is primarily educational. It is designed to help understand the workings of linear regression algorithms by building them from scratch. The implementations focus on fundamental concepts rather than on optimizing for speed or robustness, using only numpy for array processing and custom datasets for specific tasks.

This project implements the following:  
&emsp; [Ordinary Least Squares (OLS)](#ordinary-least-squares-ols-regression)  
&emsp; [Ridge Regression](#ridge-regression)  
&emsp; [Lasso Regression](#lasso-regression)   
&emsp; [RANSAC Regression](#ransac-regression)   
&emsp; [Bayesian Regression](#bayesian-regression)   
&emsp; [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)   
&emsp; [Quadratic Discriminant Analysis (QDA)](#quadratic-discriminant-analysis-qda)

## Installation

To use this project, you need to have Python installed along with the following libraries:
- numpy - for array processing and linear algebra operations
- scipy - for advanced linear algebra operations and optimization

## Ordinary Least Squares (OLS) Regression

Ordinary Least Squares (OLS) is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters to minimize the sum of the squared differences between the observed and predicted values.

### Usage Example
```python
from linearModels import OrdinaryLeastSquares
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = OrdinaryLeastSquares(fit_intercept=True)
reg.fit(X, y)

print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")
```

Output:
```
R^2 Score: 0.86
Regression Coefficients: [27.2, 45.75, 16.29, 24.21, 19.93]
Regression Intercept: -1.54
Regression Formula: y = -1.54 + 27.2017 * x_0 + 45.7470 * x_1 + 16.2942 * x_2 + 24.2104 * x_3 + 19.9321 * x_4
```


## Ridge Regression

Ridge Regression is a method of estimating the coefficients of multiple-regression models in scenarios where independent variables are highly correlated. It introduces a regularization term to the OLS cost function to shrink the coefficients.

### Usage Example
```python
from linearModels import Ridge
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = Ridge(alpha=1.0, fit_intercept=True)
reg.fit(X, y)

print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")
```

Output:
```
R^2 Score: 0.65
Regression Coefficients: [13.3, 23.05, 8.73, 12.14, 9.68]
Regression Intercept: -0.53
Regression Formula: y = -0.53 + 13.3011 * x_0 + 23.0484 * x_1 + 8.7292 * x_2 + 12.1384 * x_3 + 9.6810 * x_4
```

## Lasso Regression

Lasso Regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean. The lasso procedure encourages simple, sparse models.

### Usage Example
```python
from linearModels import Lasso
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = Lasso(alpha=1.0, fit_intercept=True)
reg.fit(X, y)

print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")
```

Output:
```
R^2 Score: 0.74
Regression Coefficients: [20.79, 35.49, 16.32, 19.58, 8.02]
Regression Intercept: 14.28
Regression Formula: y = 14.28 + 20.7884 * x_0 + 35.4879 * x_1 + 16.3211 * x_2 + 19.5828 * x_3 + 8.0185 * x_4
```

## RANSAC Regression
Performs RANSAC (Random Sample Consensus) regression to fit a model to data.

RANSAC is an iterative method to estimate parameters of a mathematical model
from a set of observed data which contains outliers. It is a non-deterministic
algorithm producing only a reasonable result with a certain probability, which
increases as more iterations are allowed. The algorithm works as follows:
1. Randomly select a subset of the original data.
2. Fit a model to this subset.
3. Determine the number of outliers by testing all the other data points against the fitted model.
4. Repeat the process for a fixed number of iterations or until a model with a sufficiently small number of outliers is found.

Gives a fitted model that best represents the inliers of the data.


### Usage Example
```python
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
```

Output: 
``` 
        No model fit, scaling threshold from 0.01 to 0.02  
        No model fit, scaling threshold from 0.02 to 0.04  
        No model fit, scaling threshold from 0.04 to 0.08  
        No model fit, scaling threshold from 0.08 to 0.16  
R^2 Score: 0.82  
Regression Coefficients: [33.79, 42.26, 13.45, 26.59, 25.77]  
Regression Intercept: 7.74  
Regression Formula: y = 7.74 + 33.7898 * x_0 + 42.2570 * x_1 + 13.4509 * x_2 + 26.5918 * x_3 + 25.7650 * x_4
```

## Bayesian Regression

Bayesian Regression is a type of regression that incorporates prior distributions on the parameters and updates these distributions based on the observed data.

### Usage Example with Auto Hyperparameter Tuning

```python
from linearModels import Bayesian
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=1000, n_features=5, noise=25, random_state=42)
reg = Bayesian(max_iter=300, tol=0.0001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, fit_intercept=True)
alpha_1, alpha_2, lambda_1, lambda_2 = reg.tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
reg.fit(X, y)

print(f"Best Hyperparameters: alpha_1={alpha_1:.2f}, alpha_2={alpha_2:.2f}, lambda_1={lambda_1:.2f}, lambda_2={lambda_2:.2f}")
print(f"R^2 Score: {r2_score(y, reg.predict(X))}")
print(f"Regression Coefficients: {reg.coef_}")
print(f"Regression Intercept: {reg.intercept_}")
print(f"Regression Formula: {reg.get_formula()}")
```

Output:
```
        Improved after, No of iterations: 0, MSE: 670.82
        Improved after, No of iterations: 1, MSE: 670.82
        Improved after, No of iterations: 2, MSE: 670.82
        ...
        Improved after, No of iterations: 48, MSE: 670.82
        Improved after, No of iterations: 49, MSE: 670.82
Stopped after 1000 iterations.
Best Hyperparameters: alpha_1=0.50, alpha_2=0.27, lambda_1=0.50, lambda_2=0.27

R^2 Score: 0.86
Regression Coefficients: [27.19 45.74 16.29 24.15 19.95]
Regression Intercept: 27.19
Regression Formula: y = 27.19 + 27.1907 * x_0 + 45.7403 * x_1 + 16.2937 * x_2 + 24.1582 * x_3 + 19.9573 * x_4
```

## Linear Discriminant Analysis (LDA)

Linear Discriminant Analysis (LDA) is a classification method that finds a linear combination of features that characterizes or separates two or more classes. It is used for dimensionality reduction while preserving as much of the class discriminatory information as possible.

### Usage Example
```python
import numpy as np
from discriminantAnalysis import LinearDiscriminantAnalysis, make_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = make_data(n_samples=1000, n_features=2, cov_class_1=np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0, cov_class_2=np.array([[0.0, -1.0], [2.5, 0.7]]).T * 2.0, shift=[4, 1], seed=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

Output:
```
Accuracy: 0.73
Classification Report:
              precision    recall  f1-score   support

         0.0       0.93      0.49      0.65       302
         1.0       0.65      0.96      0.78       298

    accuracy                           0.73       600
   macro avg       0.79      0.73      0.71       600
weighted avg       0.79      0.73      0.71       600
```

## Quadratic Discriminant Analysis (QDA)

Quadratic Discriminant Analysis (QDA) is a classification method that uses a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.

### Usage Example
```python
import numpy as np
from discriminantAnalysis import QuadraticDiscriminantAnalysis, make_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = make_data(n_samples=1000, n_features=2, cov_class_1=np.array([[0.0, -1.0], [2.5, 0.7]]) * 2.0, cov_class_2=np.array([[0.0, -1.0], [2.5, 0.7]]).T * 2.0, shift=[4, 1], seed=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = QuadraticDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
```

Output:
```
Accuracy: 0.81
Classification Report:
              precision    recall  f1-score   support

         0.0       0.88      0.74      0.80       302
         1.0       0.77      0.90      0.83       298

    accuracy                           0.81       600
   macro avg       0.82      0.82      0.81       600
weighted avg       0.82      0.81      0.81       600
```

## LDA and QDA Visualized
The plots below compare Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) using the same dataset. The ellipses represent the class conditional densities, which show the distribution of the data points for each class.

![alt text](https://raw.githubusercontent.com/SantiagoEnriqueGA/custom_linear_learning/refs/heads/main/plots/lda_qda.png)

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

# Output:
# Accuracy: 0.73
# Classification Report:
#               precision    recall  f1-score   support

#          0.0       0.93      0.49      0.65       302
#          1.0       0.65      0.96      0.78       298

#     accuracy                           0.73       600
#    macro avg       0.79      0.73      0.71       600
# weighted avg       0.79      0.73      0.71       600



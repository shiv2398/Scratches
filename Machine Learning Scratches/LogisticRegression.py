import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random

iris = load_iris()
X = iris.data
target = iris.target
X_train,X_test,y_train,y_test=train_test_split(X,target,test_size=0.2,random_state=42)

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.y_pred=[]
    
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(y_pred)
            self.y_pred.append(h)
            # Loss function
            dw = 1/n_samples * np.dot(X.T, (h - Y))
            db = 1/n_samples * np.sum(h - Y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(self.weights, X.T) + self.bias
        probs = self.sigmoid(linear_model)
        output = [0 if p < 0.5 else 1 for p in probs]
        return np.array(output)

# Example usage
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions = logreg.predict(X_test)
print("Predictions:", predictions)



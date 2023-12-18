import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from gif_making import gif_plotter
# Download the dataset
X, Y = make_regression(n_samples=500, n_features=1, noise=20, random_state=42)

# Plotting the data
plt.scatter(X, Y)
plt.title('Linear Regression Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('X train :',X_train.shape,'X test : ', X_test.shape)

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.n_iters = n_iters
        self.lr = lr
        self.bias = None
        self.weights = None
        self.y_pred=[]    

    def fit(self, X, Y):
    
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Fitting the model
            y_pred = np.dot(X, self.weights) + self.bias
            self.y_pred.append(y_pred)
            # Gradients
            dw = 1/n_samples * np.dot(X.T, (y_pred - Y))
            db = 1/n_samples * np.sum(y_pred - Y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Predict function
        return np.dot(X, self.weights) + self.bias

lr = LinearRegression(n_iters=5000)

# Training the model
print('Model Training ...')
lr.fit(X_train, y_train)

#plotting training graphs
plt.scatter(X_train,y_train)
for tr_line in lr.y_pred:
    plt.plot(X_train,tr_line)
plt.title('Liner Regression Training Graph ')
plt.xlabel('X train')
plt.ylabel('Y train')
plt.show()
# Predictions
gif_plotter(X_train,y_train,lr.y_pred)
y_pred = lr.predict(X_test)

# Plotting the values
plt.scatter(X_test, y_test, label='Actual data')
plt.plot(X_test, y_pred, label='Regression line', color='red')
plt.title('Linear Regression Predictions')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

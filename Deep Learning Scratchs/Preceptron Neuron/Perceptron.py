import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Perceptron:
    
    def __init__(self):
        self.w = None
        self.b = None
    
    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0
    
    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.model(x))
        return np.array(y_pred)
    
    def fit(self, X, Y, lr=1, epochs=1, wt=False):
        self.w = np.random.rand(X.shape[1])
        self.b = 0

        accuracy = {}
        max_accuracy = 0
        wt_matrix = []

        for i in range(epochs):
            y_pred = []
            for x_value, y_value in zip(X, Y):
                y_pred = self.model(x_value)
                
                if y_value == 1 and y_pred == 0:
                    self.w = self.w + lr * x_value
                    self.b = self.b - lr * 1
                if y_value == 0 and y_pred == 1:
                    self.w = self.w - lr * x_value
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)
            accuracy[i] = accuracy_score(self.predict(X), Y)
            
            if accuracy[i] > max_accuracy:
                max_accuracy = accuracy[i]
                chkptw = self.w.copy()
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb
        print('Maximum Accuracy:', max_accuracy)

        # Plotting accuracy
        plt.plot(list(accuracy.values()))
        plt.title('Accuracy')
        plt.ylim([0, 1])
        plt.show()

        if wt:
            return wt_matrix
        else:
            return None

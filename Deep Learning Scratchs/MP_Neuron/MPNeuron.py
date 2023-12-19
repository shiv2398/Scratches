#normal model
import numpy as np
from sklearn.metrics import accuracy_score
def I_phase_model(X,b,i):
    return 1 if np.sum(X[i,:])>=b else 0

def II_phase_model(X,y,b):
    y_pred_train=[]
    accurate_rows=0
    for x,y in zip(X,y):
        y_pred=(np.sum(x)>=b)
        y_pred_train.append(y_pred)
        accurate_rows+=(y==y_pred)
    print('Accurate Rows : ',accurate_rows,accurate_rows/X.shape[0])
    return accurate_rows

def III_phase_model(X, y):
    for b in range(X.shape[1] + 1):
        y_pred_train = []
        accurate_rows = 0.0
        #print(type(X), type(y))
        for x_value, y_value in zip(X, y):  # Use different variable names
            y_pred = (np.sum(x_value) >= b)
            y_pred_train.append(y_pred)
            accurate_rows += (y_value == y_pred)
        print(f'Threshold : {b} | Accuracy : {accurate_rows / X.shape[0] :.2f}')

class MPNeuron:

    def __init__(self):
        self.b=None

    def model(self,x):
        return (sum(x)>=self.b)
    
    def predict(self,X):
        y=[]
        for x in X:
            result=self.model(x)
            y.append(result)
        return np.array(y)
    def fit(self,X,Y):
        accuracy={}
        for b in range(X.shape[1]+1):
            self.b=b
            y_pred=self.predict(X)
            accuracy[b]=accuracy_score(y_pred,Y)
        best_b=max(accuracy,key=accuracy.get)
        self.b=best_b
        print('Optimal value of b :',best_b)
        print('Highest accuracy is :',accuracy[best_b])
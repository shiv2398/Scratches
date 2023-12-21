import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import pandas as pd 

#Sigmoid Neuron
def sigmoid_1v(x,w,b):
    return 1/(1+np.exp(-(w*x+b)))

def sigmoid_2v(x1,x2,w1,w2,b):
    return 1/1+(np.exp(-(w1*x1+w2*x2+b)))

def calculate_loss(X,Y,w_est,b_est):
    loss=0
    for x,y in zip(X,Y):
        loss+=(y-sigmoid_2v(x,w_est,b_est)**2)
    return loss

class SigmoidNeuron:
    def __init__(self):
        self.w=None
        self.b=None
    
    def perceptron(self,x):
        return np.dot(self.w,x)+self.b
    
    def sigmoid(self,X):
        return 1.0/(1.0+np.exp(-X))
    
    def grad_w(self,x,y):
        y_pred=self.sigmoid(self.perceptron(x))
        return (y_pred-y)*y_pred*(1-y_pred)*x
    
    def grad_b(self,x,y):
        y_pred=self.sigmoid(self.y_perceptron(x))
        return (y_pred-y)*y_pred*(1-y_pred)
    def fit(self,X,Y,epochs=1,learning_rate=0.1,initialise=True):

        if initialise:
            self.w=np.random.randn(1,X.shape[1])
            self.b=0
        for i in range(epochs):
            dw=0
            db=0
            for x,y in zip(X,Y):
                dw+=self.grad_w(x,y)
                db+=self.grad_b(x,y)
            self.w-=learning_rate*dw
            self.b-=learning_rate*db
    
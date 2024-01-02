import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import pandas as pd 
import matplotlib
#Sigmoid Neuron
def sigmoid_1v(x,w,b):
    return 1/(1+np.exp(-(w*x+b)))

def sigmoid_2v(x1,x2,w1,w2,b):
    return 1/(1+np.exp(-(w1*x1 + w2*x2 + b)))

def calculate_loss(X,Y,w_est,b_est):
    loss=0
    for x,y in zip(X,Y):
        loss+=(y-sigmoid_2v(x,w_est,b_est)**2)
    return loss

class SigmoidNeuron:
  
  def __init__(self):
    self.w = None
    self.b = None
    
  def perceptron(self, x):
    return np.dot(x, self.w.T) + self.b
  
  def sigmoid(self, x):
    return 1.0/(1.0 + np.exp(-x))
  
  def grad_w(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred) * x
  
  def grad_b(self, x, y):
    y_pred = self.sigmoid(self.perceptron(x))
    return (y_pred - y) * y_pred * (1 - y_pred)
  
  def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True):
    
    # initialise w, b
    if initialise:
      self.w = np.random.randn(1, X.shape[1])
      self.b = 0
    
    for i in range(epochs):
      dw = 0
      db = 0
      for x, y in zip(X, Y):
        dw += self.grad_w(x, y)
        db += self.grad_b(x, y)       
      self.w -= learning_rate * dw
      self.b -= learning_rate * db
      
if __name__ == "__main__":
#sigmoid 1 variable 
    x=1
    w=0.5
    b=0
    out=sigmoid_1v(x,0.5,0)
    print('1V sigmioid :',out)
# Sigmoid 1 variable 
    w=-1.8
    b=-0.5
    X=np.linspace(-10,10,100)
    Y=sigmoid_1v(X,w,b)

    plt.plot(X,Y)
    plt.title('Sigmoid 1v Output')
    #plt.savefig('Sigmoid V1 output ')
    plt.show()
# Sigmoid 2v output
    out=sigmoid_2v(1,0,0.5,0,0)
    print('Sigmoid 2v output : ',out)

    X1=np.linspace(-10,10,100)
    X2=np.linspace(-10,10,100)

    XX1,XX2=np.meshgrid(X1,X2)
    print(X1.shape,X2.shape,XX1.shape,XX2.shape)

    w1=2
    w2=0.5
    b=0
    Y=sigmoid_2v(XX1,XX2,w1,w2,b)
    #my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])
    plt.contourf(XX1,XX2,Y,cmap='winter_r',alpha=0.6)
    #plt.savefig('Sigmoid 2V Output')
    plt.show()
    # plotting 3d
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot_surface(XX1,XX2,Y,cmap='winter_r')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    ax.view_init(30,270)
    #plt.savefig('Sigoid 3D Output ')
    plt.show()
def calculate_loss(X, Y, w_est, b_est):
    sigmoid_values = sigmoid_1v(X, w_est, b_est)
    loss = np.sum((Y - sigmoid_values)**2)
    return loss

W = np.linspace(0, 2, 101)
B = np.linspace(-1, 1, 101)

WW, BB = np.meshgrid(W, B)

w_unknown = 0.5
b_unknown = 0.25

X = np.random.random(25) * 20 - 10
Y = sigmoid_1v(X, w_unknown, b_unknown)

Loss = np.zeros(WW.shape)

for i in range(WW.shape[0]):
  for j in range(WW.shape[1]):
    Loss[i, j] = calculate_loss(X, Y, WW[i, j], BB[i, j])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(WW, BB, Loss, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.view_init(30, 270)
#plt.savefig('Loss Sigmoid 1v in 3D')
#plt.show()

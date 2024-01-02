import sys
import os 
print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/media/sahitya/BE60ABB360AB70B7/Scratches')

from Sigmoid_Neuron import SigmoidNeuron
sn=SigmoidNeuron()
print('Toy DataSet : ')
X = np.asarray([[2.5, 2.5], [4, -1], [1, -4], [-3, 1.25], [-2, -4], [1, 5]])
Y = [1, 1, 1, 0, 0, 0]
print('X : ',X,'\nY : ', Y)
#fitting the sigmoid neuron
sn.fit(X, Y, 1, 0.25, True)
def plot_sn(X, Y, sn, ax):
  X1 = np.linspace(-10, 10, 100)
  X2 = np.linspace(-10, 10, 100)
  XX1, XX2 = np.meshgrid(X1, X2)
  YY = np.zeros(XX1.shape)
  for i in range(X2.size):
    for j in range(X1.size):
      val = np.asarray([X1[j], X2[i]])
      YY[i, j] = sn.sigmoid(sn.perceptron(val))
  ax.contourf(XX1, XX2, YY, cmap='viridis_r', alpha=0.6)
  ax.scatter(X[:,0], X[:,1],c=Y, cmap='viridis_r')
  ax.plot()
  sn.fit(X, Y, 1, 0.05, True)
N = 30
plt.figure(figsize=(10, N*5))
for i in range(N):
  print('W :   ',sn.w,'B :   ',sn.b)
  ax = plt.subplot(N, 1, i + 1)
  plot_sn(X, Y, sn, ax)
  sn.fit(X, Y, 1, 0.5, False)
  #plt.show()

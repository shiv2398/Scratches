import sys
import os 
print(os.getcwd())
from tqdm import tqdm
sys.path.append('/media/sahitya/BE60ABB360AB70B7/Scratches')
from Perceptron import Perceptron
from data import classification_data as cls 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np 
#Data
print('Importing Data .... ')
classification_data = cls.ClassificationData(sample_size=100000)
#binary data
classification_data.plot_scatter()
X_train, y_train, X_test, y_test=classification_data.train_data(1000,val=True)

print(f'Train Data : {X_train.shape} Test Data : {X_test.shape}')

#Perceptron_Model 
percep=Perceptron()
# Training
print('Training Model .... ')

wt_matrix=percep.fit(X_train,y_train,0.01,1000,True)
y_pred_test=percep.predict(X_test)
#testing on test data
acc_score=accuracy_score(y_pred_test,y_test)
print('Test Accuracy : ',acc_score)

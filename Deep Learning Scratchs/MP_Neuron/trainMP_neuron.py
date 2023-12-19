import sys
import os 
print(os.getcwd())
sys.path.append('/media/sahitya/BE60ABB360AB70B7/Scratches')
from MPNeuron import I_phase_model,II_phase_model,III_phase_model,MPNeuron
from data import classification_data as cls 
from sklearn.metrics import accuracy_score
#Data
print('Importing Data .... ')
classification_data = cls.ClassificationData(sample_size=100)
#binary data
X_train_binarised, y_train, X_test_binarised, y_test=classification_data.binarisation(1000)

# Firt I model training 
print('Training I phase Model(Static threshold One DataPoint)....')
row=5
threshold=20
output_I=I_phase_model(X_train_binarised,threshold,row)
print('Row : ',row)
print('Threshold : ',threshold)
print('Output : ',output_I)

if output_I:
    print('Model prediction : Breast Cancer is Malignant')
else:
    print('Model Prediction : Breast Cancer is Benign')

print('Training II phase Model(Static Threshold Many DataPoint)....')
threshold=20
output_II=II_phase_model(X_train_binarised,y_train,threshold)
print('Threshold : ',threshold)
print(f'Accurate rows : {output_II} On {threshold}')

print('Training III phase Model(Dynamic Threshold Many DataPoint)....')
print(X_train_binarised.shape,y_train.shape)
output_III=III_phase_model(X_train_binarised,y_train)

print('Training MP Neuron On Cancer Data...')

mp_neuron=MPNeuron()
mp_neuron.fit(X_train_binarised,y_train)
y_test_pred=mp_neuron.predict(X_test_binarised)
accuracy_score=accuracy_score(y_test_pred,y_test)
print('Accuracy : ',accuracy_score)

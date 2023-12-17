import numpy as np
def accuracy(y_pred,y_true):
    return np.sum(y_pred==y_true)/y_true.shape[0]

def confusion_matrix(y_pred, y_true):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return {
        'True Positive': TP,
        'False Positive': FP,
        'False Negative': FN,
        'True Negative': TN
    }
def precision(y_true,y_pred):
    TP=np.sum((y_true==1)&(y_pred==1))
    FP=np.sum((y_pred==0)&(y_true==1))
    return {
        'Precision':TP/(TP+FP)
    }
def Recall(y_true,y_pred):
    TP=np.sum((y_true==1)&(y_pred==1))
    FN= np.sum((y_pred==0)&(y_true==1))
    return {
        'Recall': TP/(FN+TP)
    }
def f1_score(y_true,y_pred):
    recall=Recall(y_true,y_pred)
    pre=precision(y_true,y_pred)
    r=recall['Recall']
    p=pre['Precision']
    f=2*(p*r)/(r+p)
    return {'F1-Score':f}
# Example data
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1])
y_test = np.array([1, 0, 1, 1, 0, 0, 1, 1])

# Calculate accuracy
acc = accuracy(y_true, y_test)
print(confusion_matrix(y_true,y_test))
print(Recall(y_true,y_test))
print(precision(y_true,y_test))
print(f1_score(y_true,y_test))
print("Accuracy:", acc)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd

class ClassificationData:
    def __init__(self, sample_size):
        self.sample_size = sample_size
        self.breast_cancer = load_breast_cancer()
        self.classes = None
        self.features = None
        self.target = None
        self.df = None
        self.load_data()  # Load data when the object is instantiated

    def load_data(self):
        print('Breast Cancer Dataset Loading ....')
        X = self.breast_cancer.data
        Y = self.breast_cancer.target
        self.target = Y
        self.features = X
        data = pd.DataFrame(self.breast_cancer.data, columns=self.breast_cancer.feature_names)
        data['class'] = self.breast_cancer.target
        self.df = data
        self.classes = self.breast_cancer.target_names

    def train_data(self, size, split=0.2, val=False):
        X_train, X_test, y_train, y_test = train_test_split(self.features[:size], self.target[:size], test_size=split, random_state=42)
        if val:
            return X_train, y_train, X_test, y_test
        else:
            return X_train, y_train

    def plot_scatter(self):
        plt.figure(figsize=(10, 8))
        m_c = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for class_label in np.unique(self.df['class']):
            class_data = self.df[self.df['class'] == class_label]
            plt.scatter(class_data.iloc[:, 0], class_data.iloc[:, 1], label=f'Class {self.classes[class_label]}', color=np.random.choice(m_c))

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Scatter Plot of Breast Cancer Classes')
        plt.legend()
        plt.show()
        
    def binarisation(self, threshold):
        X_train, y_train, X_test, y_test = self.train_data(size=100, split=0.2, val=True)
        X_train_binarised = np.where(X_train<threshold,0,1)
        X_test_binarised = np.where(X_test<threshold,0,1)
        return X_train_binarised, y_train, X_test_binarised, y_test
    
classification_data = ClassificationData(sample_size=100)
X_train, y_train = classification_data.train_data(size=100)
classification_data.plot_scatter()
classification_data.binarisation(1000)

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:42:45 2017

@author: Goutham
"""
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidian(a,b)

class ScrappyKNN():             #writing a classifier
    def fit(self,x_train,y_train):  #takes and features of training set as input
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self,x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_dist = euc(row,self.x_train[0])
        best_index = 0
        for i in range(1,len(self.x_train)):
            dist = euc(row,self.x_train[i])
            if dist<best_dist:
                best_dist = dist
                best_index = i
            return self.y_train[best_index]
        
        
from sklearn import datasets
iris = datasets.load_iris()

x = iris.data  #features
y = iris.target  #Labels

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5) #divide the test size to half

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = ScrappyKNN()
 
my_classifier = my_classifier.fit(x_train,y_train)

predictions = my_classifier.predict(x_test)  #input features

from sklearn.metrics import accuracy_score   #to gauge the accuracy of our predictions
print (accuracy_score(y_test,predictions))
 


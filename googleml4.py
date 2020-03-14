# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:42:45 2017

@author: Goutham
"""
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data  #features
y = iris.target  #Labels
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5) #divide the test size to half

'''
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
'''
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
 
my_classifier = my_classifier.fit(x_train,y_train)

predictions = my_classifier.predict(x_test)  #input features

from sklearn.metrics import accuracy_score   #to gauge the accuracy of our predictions
print (accuracy_score(y_test,predictions))
 


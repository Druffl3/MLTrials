import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris() #loading the iris data set, look it up on the web.
test_idx=[0,50,100]  #example id's of the labels in the data set
#training test
training_target = np.delete(iris.target,test_idx)
training_data = np.delete(iris.data,test_idx,axis=0)

#test set
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
'''
print (iris.feature_names) #features
print (iris.target_names) #label
print (iris.data[0]) #examples already in the dataset
print (iris.target[0]) #labels
for i in range(len(iris.data)): #0=sentosa, 1=versicolor, 2=virginica
    print ("Examples %d: Label: %s, Features: %s"%(i,iris.target[i],iris.data[i]))
'''
clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_data,training_target)

print ("Testing Data: ",test_target)
print ("Predicted   :",clf.predict(test_data))
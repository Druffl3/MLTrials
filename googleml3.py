from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import random

greyhounds = 50
labs = 50

grey_height = 28+4*np.random.randn(greyhounds)
labs_height = 28+4*np.random.randn(labs)

features = list(grey_height)+list(labs_height)
label = []
for i in range(100):
    label.append(random.randint(1,2)) #1=Greyhound 2=Lab

'''
plt.hist([grey_height,labs_height],stacked=True,color=['r','b'])
plt.show()
'''
print (features)
print (label)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,label)  #need more than one feature

print (clf.predict(20))



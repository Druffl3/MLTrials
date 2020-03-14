from sklearn import tree
features = [[140,1],[130,1],[150,0],[170,0]] #0=bumpy 1=smooth
label = [0,0,1,1] #0=apple 1=orange
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,label) #sklearn learning algorithm object FIT is used to train

print (clf.predict([[200,0]]))
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
# LOADING DATASETS
iris=datasets.load_iris()
# print(iris.DESCR)
# PRINTING DESCRIPTION AND FEATURES
features=iris.data
label=iris.target
# print(features[0],label[0])
# TRAINING THE CLASSIFIER
clf=KNeighborsClassifier()
clf.fit(features,label)
preds=clf.predict([[31,1,1,1]])
print(preds)



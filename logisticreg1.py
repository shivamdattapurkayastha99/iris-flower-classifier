# train a logistic regreession classifier to predict whether a flower is iris virginica or not
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris=datasets.load_iris()
# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])
# print(iris['data'].shape)
X=iris["data"][:,3:]
# print(iris["data"])
# print(X)
Y=(iris["target"]==2).astype(np.int)
# print(Y)
# print(X)
clf=LogisticRegression()
clf.fit(X,Y)
example=clf.predict(([[2.6]]))
print(example)
X_new=np.linspace(0,3,1000).reshape(-1,1)
print(X_new)
Y_prob=clf.predict_proba(X_new)
plt.plot(X_new,Y_prob[:,1],"g-",label="virginica")
plt.show()



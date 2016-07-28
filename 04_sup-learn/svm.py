# SVM in sklearn

# simple svm example from documentation
from sklearn import svm
X = [[0,0], [1,1]]
y = [0,1]
clf = svm.SVC()
clf.fit(X,y)

print(clf.predict([[2.0, 2.0]]))

# accuracy score
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

# kernels
clf = svm.SVC(kernel="linear")
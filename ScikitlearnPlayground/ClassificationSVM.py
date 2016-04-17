import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


data = np.genfromtxt('datasets/exp.csv', delimiter=',')

X = data[:, :2] 
y = data[:, [2]] 

# magic
clf = svm.SVC(kernel='rbf', C=1.0)
clf.fit(X, y)

# visualize
h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title('SVM using RBF Kernel')

plt.show()
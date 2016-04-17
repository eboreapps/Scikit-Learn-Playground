import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

# our features X1 & X2 represented by X
X = np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

# our output or target
y = [0,1,0,1,0,1]


# our svm classifer
clf = svm.SVC(kernel='linear', C=1.0)

# learn
clf.fit(X, y)

# visualize
w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()
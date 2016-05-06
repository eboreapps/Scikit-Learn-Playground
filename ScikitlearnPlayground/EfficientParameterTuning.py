#Copyright 2016 EBORE APPS (http://www.eboreapps.com)

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target

knn = KNeighborsClassifier()

k_range = range(1, 31)

param_grid = dict(n_neighbors=k_range)

grid = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy")

grid.fit(X, y)

#all scores
#print grid.grid_scores_

#first set
#print grid.grid_scores_[0].parameters
#print grid.grid_scores_[0].cv_validation_scores
#print grid.grid_scores_[0].mean_validation_score

#get only mean values
grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]

#plot
#plt.plot(k_range, grid_mean_scores)
#plt.xlabel("value of K for KNN")
#plt.ylabel("cross-validation accuracy")

#plt.show()

print grid.best_score_
print grid.best_params_
print grid.best_estimator_

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

from sklearn.grid_search import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

iris = load_iris()
weight_option = ["uniform", "distance"]

X = iris.data
y = iris.target

knn = KNeighborsClassifier()

k_range = range(1, 31)

param_dist = dict(n_neighbors=k_range, weights=weight_option)

randGSCV = RandomizedSearchCV(knn, param_dist, cv=10, n_iter=10, scoring="accuracy", random_state=5)

randGSCV.fit(X, y)

print randGSCV.best_params_
print randGSCV.best_score_
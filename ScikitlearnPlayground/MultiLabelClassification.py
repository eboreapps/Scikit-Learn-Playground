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

from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])

y_train = [[0],
		   [0],
		   [0],
		   [0],
		   [0],
		   [0],
		   [1],
		   [1],
		   [1],
		   [1],
		   [1],
		   [1],
		   [0,1],
		   [0,1]]

# Create a binary array marking values as True or False
y_train_binarized = MultiLabelBinarizer().fit_transform(y_train)

X_test = np.array(['nice day in nyc',
                   'welcome to london',
                   'hello welcome to new york. enjoy it here and london too'])   

target_names = ['New York', 'London']

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(SVC(kernel='linear')))])

classifier.fit(X_train, y_train_binarized)

predicted = classifier.predict(X_test)

for item, preds in zip(X_test, predicted):
     norm_preds = [(0 if x < 0.5 else 1) for x in preds.tolist()]
     pred_targets = ["" if x[1] == 0 else target_names[x[0]]
                     for x in enumerate(norm_preds)]
     print item, filter(lambda x: len(x.strip()) > 0, pred_targets)

#for item, labels in zip(X_test, predicted):
#	print '%s => %s' % (item, ', '.join(target_names[x] for x in labels))






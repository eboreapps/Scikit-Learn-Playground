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

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
	'UNC played Duke in basketball',
	'Duke lost the basketball game',
	'I ate a sandwich'
]

vectorizer = CountVectorizer(stop_words='english')

print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_
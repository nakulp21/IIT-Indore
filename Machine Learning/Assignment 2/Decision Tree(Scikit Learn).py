from sklearn import tree
import pandas as pd
import sklearn
headers = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']
dataset = pd.read_csv("/home/nakul/Documents/data_banknote_authentication.csv", names = headers)
inputs = dataset.drop(['Class'], axis=1)
target = dataset.iloc[:,-1]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs, target, test_size=0.2)
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)

from sklearn.tree import export_graphviz
headers = x_train.columns
export_graphviz(model, 'tree.dot', feature_names=headers)
! dot -Tpng tree.dot -o tree.png

import matplotlib.pyplot as plt
import cv2
%matplotlib inline
img = cv2.imread('tree.png')
plt.figure(figsize = (20, 20))
plt.imshow(img)

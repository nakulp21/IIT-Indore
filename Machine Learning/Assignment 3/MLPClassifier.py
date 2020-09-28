#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import pandas as pd
data = pd.read_csv("/home/nakul/Desktop/iris.csv", names=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm', 'Species'])
data.head()


# In[ ]:





# In[2]:


data = data[data.Species != 'Iris-virginica']


# In[3]:


sns.pairplot( data=data, vars=('SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'), hue='Species' )


# In[4]:


df_norm = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)


# In[5]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data["Species"] = labelencoder.fit_transform(data["Species"])


# In[30]:


inputs = data.iloc[:,:-1]


# In[33]:


target = data.iloc[:,-1]


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(df_norm, target, test_size=0.2)


# In[35]:


x_train


# In[84]:


clf = MLPClassifier(hidden_layer_sizes=(5), max_iter=500, random_state=1)
clf.fit(x_train, y_train)


# In[85]:


clf.score(x_test, y_test)


# In[86]:


p = clf.predict(x_test)
p


# In[87]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,p))
print(classification_report(y_test,p))


# In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# In[2]:


#Loading the data
headers = ['Temperature', 'Exhaust Vacuum', 'Pressure', 'Relative Humidity', 'Energy Output']
data = pd.read_excel("/home/nakul/Desktop/dataset.xlsx", names=headers)
data.head()


# In[3]:


#Data Preprocessing
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


# In[4]:


inputs = data[:,:-1]
target = data[:,-1]
inputs


# In[5]:


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs, target, test_size=0.2)


# In[6]:


x_train


# In[7]:


model = MLPRegressor(hidden_layer_sizes=(1,), solver='sgd', early_stopping=False, max_iter=1000).fit(x_train, y_train)


# In[8]:


print("{:.2%}".format(model.score(x_train, y_train)))


# In[9]:


print("{:.2%}".format(model.score(x_test, y_test)))


# In[10]:


# plot prediction and actual data
y_pred = model.predict(x_test) 
plt.plot(y_test, y_pred, '.')

# plot a line, a perfit predict would all fall on this line
x = np.linspace(-2, 2.5, 2)
y = x
plt.plot(x, y)
plt.show()


# In[11]:


print(model.coefs_)


# In[ ]:




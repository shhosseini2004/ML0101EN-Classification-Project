#!/usr/bin/env python
# coding: utf-8

# # Classification

# ### Importing required libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import scipy.optimize as opt
import itertools
from sklearn import preprocessing
from sklearn.metrics import jaccard_score, f1_score, log_loss
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load data

# In[98]:


df = pd.read_csv("heart.csv")
df.head()


# In[107]:


df.columns


# In[108]:


df["output"].value_counts()


# In[109]:


X = np.asanyarray(df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh','exng', 'oldpeak', 'slp', 'caa', 'thall']])
X[0:5]


# In[110]:


y = np.asanyarray(df[['output']])
y[0:5]


# In[111]:


print (X[0:5])
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
print (X[0:5])


# In[112]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ("Train set:", X_train.shape, y_train.shape)
print ("Test set:", X_test.shape, y_test.shape)


# In[113]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
LR


# In[114]:


yhat = LR.predict(X_test)
print (yhat)
print (y_test)


# In[127]:


print ("Jaccard_score:%.2f" % jaccard_score(y_test, yhat, pos_label=1))
print ("Log_loss: :%.2f" % log_loss(y_test, yhat_prob))
print ("F1_score: %.2f" %  f1_score(y_test, yhat, average="weighted"))


# In[119]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[129]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[0, 1])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(0)', 'Malignant(1)'], normalize=False, title='Confusion matrix')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





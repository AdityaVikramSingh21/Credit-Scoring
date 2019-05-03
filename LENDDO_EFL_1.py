
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc,classification_report
import matplotlib.pyplot as plt


# In[2]:


data_1=pd.read_csv("C:/Users/Admin/Downloads/Data Scientist - Exercises/Ex1 - Modeling sample.csv")


# In[3]:


data_1_test=data_1[data_1['sample']=='testing']
data_1_train=data_1[data_1['sample']=='training']
data_1_valid=data_1[data_1['sample']=='validation']


# In[4]:


true_labels_test=data_1_test['target']
true_labels_train=data_1_train['target']
true_labels_valid=data_1_valid['target']


# In[5]:


scores_test_1=data_1_test['model1']
scores_train_1=data_1_train['model1']
scores_valid_1=data_1_valid['model1']


# In[6]:


scores_test_2=data_1_test['model2']
scores_train_2=data_1_train['model2']
scores_valid_2=data_1_valid['model2']


# In[7]:


def plot_roc(fpr,tpr,thresholds):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

 
    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])
 
    
    plt.show()
    plt.close()


# In[8]:


# compute fpr, tpr, thresholds and roc_auc for model 1 test
fpr, tpr, thresholds = roc_curve(true_labels_test, scores_test_1)
roc_auc = auc(fpr, tpr) # compute area under the curve
plot_roc(fpr,tpr,thresholds)


# In[9]:


# compute fpr, tpr, thresholds and roc_auc for model 1 train
fpr, tpr, thresholds = roc_curve(true_labels_train, scores_train_1)
roc_auc = auc(fpr, tpr) # compute area under the curve
plot_roc(fpr,tpr,thresholds)


# In[10]:


# compute fpr, tpr, thresholds and roc_auc for model 1 valid
fpr, tpr, thresholds = roc_curve(true_labels_valid, scores_valid_1)
roc_auc = auc(fpr, tpr) # compute area under the curve
plot_roc(fpr,tpr,thresholds)


# In[11]:


# compute fpr, tpr, thresholds and roc_auc for model 2 test
fpr, tpr, thresholds = roc_curve(true_labels_test, scores_test_2)
roc_auc = auc(fpr, tpr) # compute area under the curve
plot_roc(fpr,tpr,thresholds)


# In[12]:


# compute fpr, tpr, thresholds and roc_auc for model 2 train
fpr, tpr, thresholds = roc_curve(true_labels_train, scores_train_2)
roc_auc = auc(fpr, tpr) # compute area under the curve
plot_roc(fpr,tpr,thresholds)


# In[13]:


# compute fpr, tpr, thresholds and roc_auc for model 2 valid
fpr, tpr, thresholds = roc_curve(true_labels_valid, scores_valid_2)
roc_auc = auc(fpr, tpr) # compute area under the curve
plot_roc(fpr,tpr,thresholds)


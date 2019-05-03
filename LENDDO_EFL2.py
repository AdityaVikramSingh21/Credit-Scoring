
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn import naive_bayes,linear_model,svm,model_selection,ensemble,tree,preprocessing
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


# In[2]:


data_2=pd.read_csv('C:/Users/Admin/Documents/MachineLearning/Lenddo.csv') #reading in the training set (IS fold)


# In[30]:


data_dictionary=pd.read_csv("C:/Users/Admin/Documents/MachineLearning/Data Scientist - Exercises/Data_dictionary.csv")


# In[32]:


data_dictionary=data_dictionary.iloc[3:,:] 


# In[34]:


dicti=dict(zip(data_dictionary.Var,data_dictionary.Type)) #dictionary of every feature with it's data type


# In[35]:


continous_feature=[]  #making a list of the different types of features
dummy=[]              
categoricals=[]
for k,v in dicti.items():
    if v=='Continuous':
        continous_feature.append(k)
    elif v=='Dummy':
        dummy.append(k)
    else:
        categoricals.append(k)


# In[37]:


X_continous=data_2[continous_feature]
X_categoricals=data_2[categoricals]
X_dummy=data_2[dummy]


# In[38]:


X=pd.concat([X_continous,X_categoricals,X_dummy],axis=1)


# In[40]:


x_train, x_test, y_train, y_test = train_test_split(X, data_2.iloc[:, -1], test_size=0.1, stratify=data_2.Target, random_state=4129)


# In[41]:


def make_pipe(kcount=6000, levels=[0]):
    '''Make a pipeline of an under- and an over-sampler, that produces
    kcount samples for each target classself.
    levels: a list of target classes to under-sample.'''

    ratio = dict(zip(levels, [kcount] * len(levels)))
    

    # down-sample majority classes to kcount
    rus = RandomUnderSampler(random_state=4129, ratio=ratio)

    # up-sample the others to kcount
    ros = RandomOverSampler(random_state=4129)

    from imblearn.pipeline import Pipeline
    return Pipeline([('under',rus), ('over',ros)])


# In[42]:


samp_pipe = make_pipe(6000, levels=[0])
x_train_r, y_train_r = samp_pipe.fit_sample(x_train, y_train)


# In[43]:


logit_best = linear_model.LogisticRegression(C=0.001,multi_class='multinomial',solver='newton-cg')
logit_best.fit(x_train, y_train)                     #multinomial logistic regression with cross validation on test data##unbalanced data
y_pred = logit_best.predict(x_test)
labels=[1,0]
print('accuracy of Logit Regression is: ', cross_val_score(logit_best, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit_best, x_test, y_test)), np.std(cross_val_score(logit_best, x_test, y_test))))
print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[44]:


logit_best = linear_model.LogisticRegression(C=0.001,multi_class='multinomial',solver='newton-cg')
logit_best.fit(x_train_r, y_train_r)                     #multinomial logistic regression on balanced data with cross validation on test data
y_pred = logit_best.predict(x_test)
labels=[1,0]
print('accuracy of Logit Regression on balanced data is: ', cross_val_score(logit_best, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(logit_best, x_test, y_test)), np.std(cross_val_score(logit_best, x_test, y_test))))
print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[45]:


rforest = ensemble.RandomForestClassifier(n_estimators=50,max_depth=20,random_state = 2018)
rforest.fit(x_train, y_train) #unbalanced data
y_pred = rforest.predict(x_test)
labels=[1,0]
print('accuracy of RandomForest on unbalanced data is: ', cross_val_score(rforest, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(rforest, x_test, y_test)), np.std(cross_val_score(rforest, x_test, y_test))))
print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[46]:


rforest = ensemble.RandomForestClassifier(n_estimators=50,max_depth=20,random_state = 2018)
rforest.fit(x_train_r, y_train_r) #balanced data
y_pred = rforest.predict(x_test)
labels=[1,0]
print('accuracy of RandomForest on rebalanced data is: ', cross_val_score(rforest, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(rforest, x_test, y_test)), np.std(cross_val_score(rforest, x_test, y_test))))
print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[47]:


gboost = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 2018,max_depth = 20)
gboost.fit(x_train_r, y_train_r)
y_pred = gboost.predict(x_test)
labels=[1,0]
print('accuracy of Gradboost on rebalanced data is: ', cross_val_score(gboost, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(gboost, x_test, y_test)), np.std(cross_val_score(gboost, x_test, y_test))))
print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[48]:


gboost = ensemble.GradientBoostingClassifier(n_estimators = 50, random_state = 2018,max_depth = 20)
gboost.fit(x_train, y_train)
y_pred = gboost.predict(x_test)
labels=[1,0]
print('accuracy of Gradboost on unbalanced data is: ', cross_val_score(gboost, x_test, y_test)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(gboost, x_test, y_test)), np.std(cross_val_score(gboost, x_test, y_test))))
print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[49]:


adaboost = [
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME', random_state = 2018),
    ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 20), n_estimators = 50, algorithm ='SAMME.R', random_state = 2018)]
for i in range(2):
    adaboost[i].fit(x_train, y_train)
    y_pred = adaboost[i].predict(x_test)
    labels=[1,0]
    print('accuracy of adaboost on unbalanced data is: ', cross_val_score(adaboost[i], x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(adaboost[i], x_test, y_test)), np.std(cross_val_score(adaboost[i], x_test, y_test))))
    print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
    print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[51]:


for i in range(2):
    adaboost[i].fit(x_train_r, y_train_r)
    y_pred = adaboost[i].predict(x_test)
    labels=[1,0]
    print('accuracy of adaboost on rebalanced data is: ', cross_val_score(adaboost[i], x_test, y_test)) 
    print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(adaboost[i], x_test, y_test)), np.std(cross_val_score(adaboost[i], x_test, y_test))))
    print('precision',precision_score(y_test, y_pred, average=None, labels=labels))
    print('recall',recall_score(y_test, y_pred, average=None, labels=labels))


# In[24]:


#reading the given test dataset(OS)
OS_data=pd.read_csv("C:/Users/Admin/Documents/MachineLearning/Data Scientist - Exercises/LENDDO_EFL_OS.csv")
x_test_OS=OS_data.iloc[:,1:-1]
y_test_OS=OS_data.iloc[:,-1]


# In[53]:


#checking the best model on the provided test dataset
y_pred_OS = gboost.predict(x_test_OS)
labels=[1,0]
print('accuracy of Best Model(Gboost) on OS data is: ', cross_val_score(gboost, x_test_OS, y_test_OS)) 
print('mean: %.3f, standard deviation: %.3f' % (np.mean(cross_val_score(gboost, x_test_OS, y_test_OS)), np.std(cross_val_score(gboost, x_test_OS, y_test_OS))))
print('precision',precision_score(y_test_OS, y_pred_OS, average=None, labels=labels))
print('recall',recall_score(y_test_OS, y_pred_OS, average=None, labels=labels))


# In[54]:


OS_data['predictions']=y_pred_OS


# In[56]:


OS_data.to_csv('OS_data_withPredictions.csv')


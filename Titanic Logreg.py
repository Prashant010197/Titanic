# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:35:46 2020

@author: hp
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, log_loss, confusion_matrix

def scores(Y_test, y_pred):
    print('Accuracy Score is:{}'.format(accuracy_score(Y_test, y_pred)))
    print('F1 Score is:{}'.format(f1_score(Y_test, y_pred)))
    print('Sensitivity Score is:{}'.format(recall_score(Y_test, y_pred)))
    print('Precision Score is:{}'.format(precision_score(Y_test, y_pred)))
    print('ROC AUC Score is:{}'.format(roc_auc_score(Y_test, y_pred)))

''' Pickled Dataframe read into df1'''
df=pd.read_pickle('Cleaned_data')

df=pd.read_csv('train.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)

sex=pd.get_dummies(df['Sex'], drop_first=True)
embark=pd.get_dummies(df['Embarked'], drop_first=True)

df=pd.concat([df,sex,embark],axis=1)

df.drop(labels=['Sex','Embarked','Cabin','PassengerId','Ticket','Name'], inplace=True, axis=1)

'''UPSAMPLING'''
df_minority = df[df.Survived==1]
df_majority = df[df.Survived==0]

from sklearn.utils import resample
 

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=549,
                                 random_state=123)
 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df=df_upsampled


''' Train.csv Data split into test and train set'''
X=df.iloc[:,1:].values
Y=df.iloc[:,0].values

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=42, test_size=0.2)


'''LOGREG: BASELINE MODEL'''

classifier=LogisticRegression(solver='lbfgs', random_state = 1, max_iter=1000, C=10000)

''' lbfgs among other algorithms is used for multiclass problems. liblinear is applicable
only for one versus rest classification. Increased maximum iterations to 1000 to ensure
convergence'''

''' Classifier made to learn the training data'''
classifier.fit(X_train, Y_train)

''' y_pred holds values of predicted Y_test values using the X_test feature values'''
y_pred = classifier.predict(X_test)
scores(Y_test, y_pred)

'''LOGREG: Hyperparameter tuning with GRIDSEARCH'''

tuned = [{'C': [10**-4, 10**-2, 10**0, 10**2, 10**4]}]

model = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=1000), tuned, scoring = 'accuracy', cv=5)

model.fit(X_train, Y_train)

print(model.best_estimator_)
print(model.score(X_test, Y_test))
model.best_params_

classifier=LogisticRegression(solver='lbfgs', random_state = 1, max_iter=1000, C=1)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

scores(Y_test, y_pred)

'''Before tuning
Accuracy Score is:0.8272727272727273
F1 Score is:0.8061224489795918
Sensitivity Score is:0.7821782178217822
Precision Score is:0.8315789473684211
ROC AUC Score is:0.8238622181545886'''

'''After tuning
Accuracy Score is:0.8272727272727273
F1 Score is:0.8061224489795918
Sensitivity Score is:0.7821782178217822
Precision Score is:0.8315789473684211
ROC AUC Score is:0.8238622181545886'''














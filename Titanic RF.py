# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:03:13 2020

@author: hp
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from sklearn.utils import resample
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score, precision_score, recall_score, log_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def scores(Y_test, y_pred):
    print('Accuracy Score is:{}'.format(accuracy_score(Y_test, y_pred)))
    print('F1 Score is:{}'.format(f1_score(Y_test, y_pred)))
    print('Sensitivity Score is:{}'.format(recall_score(Y_test, y_pred)))
    print('Precision Score is:{}'.format(precision_score(Y_test, y_pred)))
    print('ROC AUC Score is:{}'.format(roc_auc_score(Y_test, y_pred)))

df=pd.read_csv('train.csv')

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

relative=[]
df['Relatives']=df['Parch']+df['SibSp']

for data in range(len(df)):
    if df['Relatives'].iloc[data]==0:
        relative.append('Y')
    else:
        relative.append('N')
df['Travelled Alone']=relative

df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
features=["Pclass", "Sex", "SibSp", "Parch",'Embarked','Travelled Alone','Relatives']
dummy=pd.get_dummies(df[features])
df=df[['Survived','Age', 'Fare']]
df=pd.concat([df, dummy], axis=1)

'''
train_numerical_features = list(df.select_dtypes(include=['int64', 'float64', 'int32']).columns)
ss_scaler = StandardScaler()
train_data_ss = pd.DataFrame(data = df)
train_data_ss[train_numerical_features] = ss_scaler.fit_transform(train_data_ss[train_numerical_features])
'''

features=["Pclass", "Sex", "SibSp", "Parch"]
dummy=pd.get_dummies(df[features])
X=X[['Pclass', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']].astype(int)
Y=df['Survived'].astype(int)

X=df.iloc[:,1:].values
Y=df.iloc[:,0].values

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=42, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(bootstrap= True, max_depth= 90, max_features= 3, min_samples_leaf= 5, min_samples_split= 10, n_estimators= 100)
forest.fit(X_train, Y_train)
y_pred=forest.predict(X_test)

param_test1 = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
  
grid = GridSearchCV(RandomForestClassifier(), param_test1, refit = True) 
grid.fit(X_train, Y_train)
print(grid.score(X_test, Y_test))
print(grid.best_params_)
print(grid.best_estimator_)

'''XGB'''
import xgboost as xgb

grid = xgb.XGBClassifier(colsample_bytree= 0.5, gamma=0.4, learning_rate=0.05, max_depth=3, min_child_weight=5)
grid.fit(X_train, Y_train)
y_pred=grid.predict(X_test)

scores(Y_test, y_pred)

param_grid={"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]}
grid = GridSearchCV(xgb.XGBClassifier(), param_grid, refit = True, scoring='accuracy') 
grid.fit(X_train, Y_train)
print(grid.best_estimator_)
print(grid.best_params_)
print(grid.score(X_test, Y_test))








df2=pd.read_csv('test.csv')

df2=df1

df1['Age'].fillna(df1['Age'].mean(), inplace=True)

df1['Fare'].fillna(df1['Fare'].median(), inplace=True)

df1.drop(labels=['Cabin','PassengerId','Ticket','Name'], inplace=True, axis=1)

relative1=[]

df1['Relatives']=df1['Parch']+df1['SibSp']

for data in range(len(df1)):
    if df1['Relatives'].iloc[data]==0:
        relative1.append('Y')
    else:
        relative1.append('N')

df1['Travelled Alone']=relative1


features=["Pclass", "Sex", "SibSp", "Parch",'Embarked','Travelled Alone','Relatives']

dummy=pd.get_dummies(df1[features])

df1=df1[['Age', 'Fare']]

df1=pd.concat([df1, dummy], axis=1)

X=pd.get_dummies(df1[features])

df1.drop(['PassengerId'],axis=1, inplace=True)

df1=pd.concat([df1,X],axis=1)

df1.drop(labels=['Sex','Embarked'], inplace=True, axis=1)

y_pred=grid.predict(df1)

df_ans=pd.DataFrame(y_pred)

df_ans.columns=['Survived']
df_ans['PassengerId']=df2['PassengerId']
df_ans.set_index('PassengerId', inplace=True)
df_ans.index.name='PassengerId'
df_ans.to_csv('Predicted Labelsrftune.csv')

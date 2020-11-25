# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:35:02 2020

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

def scores(Y_test, y_pred):
    print('Accuracy Score is:{}'.format(accuracy_score(Y_test, y_pred)))
    print('F1 Score is:{}'.format(f1_score(Y_test, y_pred)))
    print('Sensitivity Score is:{}'.format(recall_score(Y_test, y_pred)))
    print('Precision Score is:{}'.format(precision_score(Y_test, y_pred)))
    print('ROC AUC Score is:{}'.format(roc_auc_score(Y_test, y_pred)))

df=pd.read_pickle('Cleaned_data')
df=pd.read_csv('train.csv')

'''UPSAMPLING'''

df_minority = df[df.Survived==1]
df_majority = df[df.Survived==0]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=549,
                                 random_state=123)
 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

df=df_upsampled

X=df.iloc[:,1:].values
Y=df.iloc[:,-10].values

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, random_state=42, test_size=0.2)

df['Survived'].value_counts()/df.shape[0]

classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
scores(Y_test,y_pred)


from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)
forest.fit(X_train, Y_train)
y_pred=forest.predict(X_test)

scores(Y_test,y_pred)

'''Filling in prediction values in test file'''
df2=pd.read_csv('test.csv')

df2=df1

df1['Age'].fillna(df1['Age'].mean(), inplace=True)

df1['Fare'].fillna(df1['Fare'].median(), inplace=True)

df1.drop(labels=['Cabin','PassengerId','Ticket','Name'], inplace=True, axis=1)

sex=pd.get_dummies(df1['Sex'], drop_first=True)
embark=pd.get_dummies(df1['Embarked'], drop_first=True)

df1=pd.concat([df1,sex,embark],axis=1)

df1.drop(labels=['Sex','Embarked'], inplace=True, axis=1)

y_pred=classifier.predict(df1)

df_ans=pd.DataFrame(y_pred)

df_ans.columns=['Survived']
df_ans['PassengerId']=df2['PassengerId']
df_ans.set_index('PassengerId', inplace=True)
df_ans.index.name='PassengerId'
df_ans.to_csv('Predicted Labels.csv')


'''Dimensionality Reduction'''

from sklearn.ensemble import RandomForestRegressor
df=df.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
model = RandomForestRegressor(random_state=1, max_depth=10)
df=pd.get_dummies(df)
model.fit(df,train.Item_Outlet_Sales)

features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-9:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

from sklearn.feature_selection import SelectFromModel
feature = SelectFromModel(model)
Fit = feature.fit_transform(df, train.Item_Outlet_Sales)



'''RF no tuning scores
scores(Y_test,y_pred)
Accuracy Score is:0.8954545454545455
F1 Score is:0.8909952606635071
Sensitivity Score is:0.9306930693069307
Precision Score is:0.8545454545454545
ROC AUC Score is:0.8981196438971628'''
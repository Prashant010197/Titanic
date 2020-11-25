# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 22:29:41 2020

@author: hp
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv('train.csv')

df.shape

''' Data has 891 rows and 12 features'''

df.columns

df.duplicated(subset='Ticket').value_counts()

df.drop_duplicates(subset='Ticket', inplace=True)


'''
PassengerId=id
Survived=0 for no and 1 for yes
Pclass= Ticket Class(1st, 2nd, 3rd)
Name= Name
Sex=Sex
Age=Age
Sibsp=no. of siblings/spouses onboard
parch= number of parents/children onboard
ticket=ticket number
fare=Passenger fare
cabin=Cabin Number
Embarked=Port of embarkation: C=Cherbourg, Q=Queenstow(Cobh, Ireland currently)n, S=Southampton
'''
emb={'C':'Cherbourg', 'Q':'Cobh', 'S':'Southampton'}

df.Embarked.replace(emb, inplace=True)

tier={1:'1st class',2:'2nd class',3:'3rd class'}

df.Pclass.replace(tier, inplace=True)

df.to_excel('Simplified titanic.xlsx')

''' Features Age, Cabin and Embarked have null values. Imputation can be done in case 
of Age but categorical variables cannot be imputed and will be dropped.'''

'''UNIVARIATE EDA'''


df['Survived'].value_counts()
Out[9]: 
0    549
1    342

''' 61% of the people onboard did not survive while only 39% of the people survived'''

df.Pclass.value_counts()
Out[12]: 
3    491
1    216
2    184

c=df.groupby('Pclass')['Pclass'].count()
sns.barplot(x=c.index, y=c.values)
plt.title('Tickets sold per Class')
plt.savefig('Barplot Tickets sold per Class.png')

plt.pie(c, labels=c.index, autopct='%.2f', shadow=True)
plt.title('Tickets sold per Class')
plt.savefig('Pieplot Tickets sold per Class.png')


''' Assuming that all spaces onboard were sold, and that the total number of space was
891, 1st class had 216, 2nd class had 184 and 3rd class had 491 seats/beds'''

df.Sex.value_counts()
Out[18]: 
male      577
female    314

c=df.groupby('Sex')['Sex'].count()
sns.barplot(x=c.index, y=c.values)
plt.title('Total people by Gender')
plt.savefig('Barplot Genderwise total people.png')

plt.pie(c, labels=c.index, autopct='%.2f')
plt.title('Total people by Gender')
plt.savefig('Pieplot Genderwise total people.png')

''' Only 35% of people were female while 65% were male'''

sns.boxplot(df['Age'])
plt.title('titanic age boxplot')
plt.savefig('titanic age boxplot(without imputation')

df['Age'].fillna(df['Age'].median(), inplace=True)

sns.distplot(df['Age'])
plt.title('Distribution of Age with median as imputed values')
plt.savefig('titanic age distplot')

sns.boxplot(df['Age'])
plt.title('titanic age boxplot')
plt.savefig('titanic age boxplot(with imputation')

sns.violinplot(x=df.Age, data=df)
plt.savefig('titanic violin plot.png')

plt.rcParams['figure.figsize'] = [10, 6]
fig, axs=plt.subplots(nrows=4)
sns.distplot(df['Age'], ax=axs[0])
sns.boxplot(df['Age'], ax=axs[1])
sns.violinplot(df['Age'], ax=axs[2])
sns.swarmplot(df['Age'], ax=axs[3])
plt.savefig('Distributions after imputation.png')

df['Age'].value_counts().head()
Out[40]: 
24.0    30
22.0    27
18.0    26
19.0    25
30.0    25


''' Most of the ages fall between 20 and 40 with some outliers going upwards of
65. Missing values of Age were imputed with median as outliers are present. Most frequent
Age comes up to be 24.'''

df.SibSp.sum()
df.Parch.sum()

''' 466 siblings/spouses were onboard while 340 were parents/children'''

df.Fare.value_counts()

''' Value 0.00 is counted 15 times in the Fare feature. Could be result of improper
data handling. Could be freeloaders who entered the ship without a ticket'''

sns.distplot(df['Fare'])
sns.boxplot(df['Fare'])
plt.title('Fare boxplot')
plt.savefig('Fare boxplot')

max(df['Fare'])


'''Large number of outliers exist in Fare. Could be because only a few people 
went on a longer route or had enough money to spend, maximum fare was 512.32'''

sns.distplot(df['Embarked'].value_counts())

Out[79]: 
S    644
C    168
Q     77

c=df.groupby('Embarked')['Embarked'].count()
sns.barplot(x=c.index, y=c.values)
plt.title('Embarked Distribution')
plt.savefig('Barplot Embarked Distribution')

plt.pie(c, labels=c.index, autopct='%.2f')
plt.title('Embarked Distribution')
plt.savefig('Pieplot Embarked Distribution')

'''Over 72% of people embarked from Southampton'''



''' MULTIVARIATE EDA'''

c=df.corr()

'''There exists no strong linear correlation between the features'''

c=df.groupby('Sex')['Survived'].sum()
''' Grouped using Sex, apply count() on column Survived'''
sns.barplot(x=c.index, y=c.values)
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Sex vs Survived')
plt.savefig('Barplot Sex vs Survived.png')

plt.pie(c, labels=c.index)
plt.title('Sex vs Survived')
plt.savefig('Pieplot Sex vs Survived')
    
''' Summation of Survived values while grouping the data against Gender reveals that
a total of 342 people, 38% of people survived the disaster. 233(68%) of those were female and 
109(32%) were male. Even with majority of passengers being male(577 or 64%), the surviving
majority is female. There is evidence of a priority that would have been given to saving the 
lives women and children.'''

plt.rcParams['figure.figsize'] = [10, 9]

from numpy import median
c=df.groupby('Sex')['Age'].median()
sns.barplot(x=c.index, y=c.values)
sns.barplot(x=df['Sex'], y=df['Age'], estimator=median)
plt.title('Age vs Sex')
plt.savefig('Barplot Age vs Sex.png')

plt.pie(c, labels=c.index)
plt.title('Age vs Sex')
plt.savefig('Pieplot Age vs Sex.png')

''' Most people(male and female) were in their late 20s when they boarded the ship'''

c=df.groupby('Pclass')['Age'].median()
sns.barplot(x=c.index, y=c.values)

sns.barplot(x='Pclass', y='Age', data=df, estimator=median)
plt.title('Age vs Pclass')
plt.savefig('Barplot Age vs Pclass.png')

''' Bar plot shows that the younger people were in 3rd while the more older
people were in 1st class. This indicates that the older people had more money 
to spend than people in their 20s.'''

c=df.groupby('Survived')['SibSp'].sum()
sns.barplot(x=c.index, y=c.values)


c=df.groupby('Survived')['Parch'].sum()
sns.barplot(x=c.index, y=c.values)

'''Survival would have depended upon number of siblings, parents, Class, Age, Sex,Country of 
Embarking'''

''' Data Wrangling for Classification'''

df['Age'].fillna(df['Age'].median(), inplace=True)
df.isnull().sum()
df.drop(labels=['Cabin','PassengerId','Ticket','Name'], inplace=True, axis=1)
df.dropna(inplace=True)

sex=pd.get_dummies(df['Sex'], drop_first=True)
embark=pd.get_dummies(df['Embarked'], drop_first=True)
pcl=pd.get_dummies(df['Pclass'], drop_first=True)

df=pd.concat([df,sex,embark,pcl],axis=1)

df.drop(labels=['Sex','Pclass','Embarked'], inplace=True, axis=1)

df.to_pickle(path='Cleaned_data')























# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 16:19:32 2020

@author: shiv0
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv("Diabetes-db.csv")
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

X[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=X[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
X[['DiabetesPedigreeFunction']]=X[['DiabetesPedigreeFunction']].replace(0,np.nan)

X['Pregnancies'].fillna(0,inplace=True)
X['Glucose'].fillna(X['Glucose'].median(),inplace=True)
X['BloodPressure'].fillna(X['BloodPressure'].median(),inplace=True)
X['SkinThickness'].fillna(X['SkinThickness'].mean(),inplace=True)
X['Insulin'].fillna(X['Insulin'].mean(),inplace=True)
X['BMI'].fillna(X['BMI'].mean(),inplace=True)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#Logistic Regression
# Accuracy=80.20/82.46

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
'''
#KNN with n_neighbors=5 Accuracy is 80.20
#kNN with n_neighbors=6 Accuracy is 79.16
#KNN with n_neighbors=7 Accuracy is 77.08
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
'''

'''
#SVM with accuracy 80.20
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
'''

'''
#KERNAL SVM with accuracy 77.60
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)
'''

'''
#naive bayes with accuracy 76.56
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)
'''

'''
#Desicion Tree with accuracy 77.60
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier.fit(X_train,y_train)
'''

'''
#Random Forest with accuracy 72.39
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
'''

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix , accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))
import numpy as np
import pandas as pd
import sys

data = pd.read_csv('OpportunityScoreData.csv', nrows=100)
#print(data.loc[(data['CustomerID'] == 98761259) & (data['category'] == 'Hilti Services')])
#sys.exit()
#print(data.describe())

#preprocess data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

le.fit(data['BusinessPosition'])
transformedBusinessPosition = le.transform(data['BusinessPosition'])
data['transformedBusinessPosition'] = transformedBusinessPosition

le.fit(data['DateTime'])
transformedDateTime = le.transform(data['DateTime'])
data['transformedDateTime'] = transformedDateTime

le.fit(data['category'])
transformedCategory = le.transform(data['category'])
data['transformedCategory'] = transformedCategory


#declare features and label
feature_names = ['BookToMarket', 'BusinessType', 'Businessvolume', 'CustomerID', 'EquipmentPrice', 'HoursUsage', 'Marketcap', 'MonthlyMaintenance', 'Noofdevices', 'NumberOfEmployees', 'ProductWeight', 'Profit', 'TotalRevenue', 'Rank', 'transformedBusinessPosition', 'transformedDateTime', 'transformedCategory']
X = data[feature_names]
y = data['Rank']
#print(X)
#print(y)

#Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#print(y_test)

#now parse user input for test data
customerID = request['customerID']
category = request['category']
testData = data.loc[(data['CustomerID'] == customerID) & (data['category'] == category)]
X_test = testData[feature_names]
y_test = testData['Rank']


#apply scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### generate models with diff algos

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)), end="\n\n")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)), end="\n\n")
#Setting max decision tree depth to help avoid overfitting
clf2 = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf2.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf2.score(X_test, y_test)), end="\n\n")     

#Apply K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}' . format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}' . format(knn.score(X_test, y_test)), end="\n\n")

#Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(X_test, y_test)), end="\n\n")

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)), end="\n\n")

#Support Vector Machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)), end="\n\n")
     

# check confusion matrix and classification report (here for )
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

pred = clf2.predict(X_test)
print(pred)
#print(confusion_matrix(y_test, pred))
#print(classification_report(y_test, pred))


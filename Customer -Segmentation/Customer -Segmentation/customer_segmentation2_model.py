import os 
import json
import numpy as np
import pandas as pd
import dill as pickle

def build_and_train():
    data = pd.read_csv('customer_segmentation_data_manupulated.csv')
    rows1 = [i for i,x in enumerate(data.iloc[:,1]) if data.iloc[i,1] < 'Jun-18' ]
    rows2 = [i for i,x in enumerate(data.iloc[:,1]) if data.iloc[i,1] == 'Jun-18' ]
    
    testColumns = "CustomerID,DateTime,BookToMarket,BusinessType,Businessvolume,EquipmentPrice,HoursUsage,Marketcap,MonthlyMaintenance,Noofdevices,NumberOfEmployees,ProductWeight,Profit,TotalRevenue,category,BusinessPosition,cluster,score"
    
    traindata = data.iloc[rows1,:]
    testdata= data.iloc[rows2,:]

    import numpy as np
    np.savetxt("customer_segmentation2_traindata.csv", traindata, header=testColumns, delimiter=",", fmt='%s', comments="")
    np.savetxt("customer_segmentation2_testdata.csv", testdata, header=testColumns, delimiter=",", fmt='%s', comments="")
    
    #print(data.describe())
    
    #declare features and label
    feature_names = ['BookToMarket', 'BusinessType', 'Businessvolume', 'EquipmentPrice', 'HoursUsage', 'Marketcap', 'MonthlyMaintenance', 'Noofdevices', 'NumberOfEmployees', 'ProductWeight', 'Profit', 'TotalRevenue', 'category', 'BusinessPosition', 'cluster']
    X_train = traindata[feature_names]
    y_train = traindata['score']
    
    X_test = testdata[feature_names]
    y_test = testdata['score']
    #print(X)
    #print(y)
    
    #Create training and test sets
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\
    
    #print(y_test)
    
    #apply scaling
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    ### generate models with diff algos
    
    #Linear Regression
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(linreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(linreg.score(X_test, y_test)), end="\n\n")
    
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
    
    pred = logreg.predict(X_test)
    print(pred)
    #print(confusion_matrix(y_test, pred))
    #print(classification_report(y_test, pred))
    
    return(logreg)
    
if __name__ == '__main__':
    model = build_and_train()

    filename = 'model_customer_segmentation1.pk'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


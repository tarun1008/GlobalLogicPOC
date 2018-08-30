import pandas as pd
import numpy as np
import json
import dill as pickle
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    data = pd.read_csv('customer_segmentation2_testdata.csv')    
    
    try:
        #parse user input for test data
        test_json = request.get_json(silent=True)
        date = test_json['month']
        testData = data.loc[data['DateTime'] == date]
        
        feature_names = ['BookToMarket', 'BusinessType', 'Businessvolume', 'EquipmentPrice', 'HoursUsage', 'Marketcap', 'MonthlyMaintenance', 'Noofdevices', 'NumberOfEmployees', 'ProductWeight', 'Profit', 'TotalRevenue', 'category', 'BusinessPosition', 'cluster']
        
        X_test = testData[feature_names]
        y_test = testData['score']
        #print(X_test)
        
        
        #apply scaling
        #from sklearn.preprocessing import MinMaxScaler
        #scaler = MinMaxScaler()
        #X_test = scaler.fit_transform(X_test)
        #X_test = scaler.transform(X_test)

        #print(y_test)
    except Exception as e:
        raise e

    clf = 'model_customer_segmentation1.pk'

    print("Loading the model...")
    loaded_model = None
    with open(clf,'rb') as f:
        loaded_model = pickle.load(f)

    print("The model has been loaded...doing predictions now...")
    predictions = loaded_model.predict(X_test)
    #print(predictions)

    prediction_series = list(pd.Series(predictions))
    testData = testData.assign(score=prediction_series)

    processData = testData[['CustomerID', 'score']]    
    processDataMean = processData.groupby('CustomerID')['score'].mean().reset_index()
    processDataMean['rank'] = processDataMean['score'].rank(ascending=False, method='first')
    
    returnDataMean = processDataMean
    
    if test_json.get('customerIds'):            
        customerID = test_json['customerIds']
        returnDataMean = processDataMean.loc[processDataMean['CustomerID'].isin(customerID)]

    
    returnObj = returnDataMean
    
    resp = Response(response=returnObj.to_json(orient="records"), status=200, mimetype="application/json")
    return(resp)
        
if __name__ == '__main__':
    app.run(host='localhost', debug=False, use_reloader=True)
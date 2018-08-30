import pandas as pd
import json
import dill as pickle
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    data = pd.read_csv('testdata.csv')    
    try:
        #parse user input for test data
        test_json = request.get_json(silent=True)
        
        customerID = test_json['customerID']
        #category = test_json['category']
        date = test_json['date']
        testData = data.loc[(data['CustomerID'] == customerID) & (data['DateTime'] == date)]
        feature_names = ['BookToMarket', 'Businessvolume', 'EquipmentPrice', 'HoursUsage', 'Marketcap', 'MonthlyMaintenance', 'Noofdevices', 'NumberOfEmployees', 'ProductWeight', 'Profit', 'TotalRevenue', 'BusinessPosition_Stable', 'BusinessPosition_Growth', 'BusinessPosition_Decline', 'BusinessType_1', 'BusinessType_2', 'BusinessType_3', 'BusinessType_4', 'category_Power_Tools', 'category_Hilti_Services', 'category_Contractor_Services', 'category_Fastners_Firestop_Strut']
        
        X_test = testData[feature_names]
        y_test = testData['Rank']
        
        #print(X_test)
        #print(y_test)
    except Exception as e:
        raise e

    clf = 'model_v1.pk'

    if X_test.empty:
        return(bad_request())
    else:
        print("Loading the model...")
        loaded_model = None
        with open(clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(X_test)
        #print(predictions)

        prediction_series = list(pd.Series(predictions))
        testData = testData.assign(prediction=prediction_series)
        #print(testData)
        returnFeatures = ['CustomerID', 'DateTime', 'category', 'Rank', 'prediction']
        returnData = testData[returnFeatures]
        
        resp = Response(response=returnData.to_json(orient="records"), status=200, mimetype="application/json")
        return(resp)
        
if __name__ == '__main__':
    app.run(host='localhost', debug=False, use_reloader=True)
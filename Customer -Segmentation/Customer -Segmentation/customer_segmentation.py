import numpy as np
import pandas as pd
import csv
import json
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder

try:
    data1 = pd.read_csv("CustomerSegmentationData.csv")
    data = data1.iloc[:,2:16]
    #drop unwanted columns.
    #data.drop(['CustomerID','DateTime'], axis = 1, inplace= True)
    print("Customer Segmentation Data dataset has {} samples.".format(data.shape))
except:
    print("dataset not loaded.")

data_dumies = pd.get_dummies(data)

#Display a description of dataset.
stats = data_dumies.describe()

#scaling the data.
scaler = MinMaxScaler()
scaled_data= scaler.fit_transform(data_dumies)

#fitting hirearchical clustering to dataset
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'ward')
y_hc = hc.fit_predict(scaled_data)
print(y_hc)

#converting numpy.ndarray to list
hc_value = y_hc.tolist()

#create new df 
df = pd.DataFrame({'cluster':hc_value})

new_data = data1
#new_data.drop(['BookToMarket','BusinessPosition','BusinessType','Businessvolume','EquipmentPrice','HoursUsage', 'Marketcap', 'MonthlyMaintenance', 'Noofdevices', 'NumberOfEmployees', 'ProductWeight', 'Profit', 'TotalRevenue', 'category'], axis = 1, inplace= True)

new_data=pd.concat([new_data,df],axis=1)
#np.savetxt("customer_segmentation_data_manupulated.csv", new_data, delimiter=",", fmt='%s', header=','.join(list(new_data)), comments="")

new_data1=new_data[new_data.DateTime=='Jun-18']
cluster0 = new_data1[new_data1['cluster']==0]
cluster0 = np.unique(cluster0['CustomerID'].tolist())
print(len(cluster0))

cluster1 = new_data1[new_data1['cluster']==1]
cluster1 = np.unique(cluster1['CustomerID'].tolist())
print(len(cluster1))

cluster2 = new_data1[new_data1['cluster']==2]
cluster2 = np.unique(cluster2['CustomerID'].tolist())
print(len(cluster2))

data2 ={}
data2["Months"] = "June-18"
data2["levels"] = ["Low","Medium","High"]
data2["value"] = [len(cluster0),len(cluster1),len(cluster2)]
data2["value0"] = cluster0.tolist()
data2["value1"] = cluster1.tolist()
data2["value2"] = cluster2.tolist()
	

# Writing JSON data into a file called JSONData.json
with open('JSONData.json', 'w') as f:
     json.dump(data2, f)



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
    data = pd.read_csv("CustomerSegmentationData.csv")
    #drop unwanted columns.
    data.drop(['CustomerID','DateTime'], axis = 1, inplace= True)
    print("Customer Segmentation Data dataset has {} samples.".format(data.shape))
except:
    print("dataset not loaded.")

print(type(data))

print(list(data.columns))

data_dumies = pd.get_dummies(data)
print(list(data_dumies.columns))

#data = pd.concat([data,pd.get_dummies(data)],axis = 1)
#data = data.drop([],axis = 1, inplace = True)
print(data_dumies.head())

#Display a description of dataset.
stats = data_dumies.describe()
print(stats)

#scaling the data.
scaler = MinMaxScaler()
print(scaler.fit(data_dumies))
scaled_data= scaler.fit_transform(data_dumies)
print(scaled_data)

#applying PCA
pca = PCA(n_components=None)
data_new = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)
#output[0.11870277  0.11186081  0.10787617  0.09865675  0.09658614  0.09457023 0.0935149   0.09306136  0.09076613  0.08002097  0.01438376]

#fitting KMeans to dataset
print("kMeans clustering")
kmeans = KMeans(n_clusters=3 , init='k-means++', max_iter= 300 , n_init= 10,random_state=0)
y_kmeans = kmeans.fit_predict(scaled_data)
print(y_kmeans)
print(type(y_kmeans))  #<class 'numpy.ndarray'>

#converting numpy.ndarray to list
kmeans_value = y_kmeans.tolist()
print(type(kmeans_value))

#fitting hirearchical clustering to dataset
print("hirearchical clustering")
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage= 'ward')
y_hc = hc.fit_predict(scaled_data)
print(y_hc)
print(type(y_hc)) #<class 'numpy.ndarray'>

#converting numpy.ndarray to list
hc_value = y_hc.tolist()
print(type(hc_value))

#counting number of Customer in each cluster
from collections import Counter
counter = Counter(kmeans_value)
kmeans_outcomes=[list(counter.keys()),list(counter.values())]
print(kmeans_outcomes)

counter1 = Counter(hc_value)
hc_outcomes=[list(counter1.keys()),list(counter1.values())]
print(kmeans_outcomes)


data = {
    "Months":"June-18",
	"value1":
	"value2":
	"value3":

 
}

#json.dumps() method turns a Python data structure into JSON:
jsonData = json.dumps(data)
print(jsonData)

# Writing JSON data into a file called JSONData.json
#Use the method called json.dump()
#It's just dump() and not dumps()
#Encode JSON data
with open('JSONData.json', 'w') as f:
     json.dump(jsonData, f)


"""
#csvfile = "<path to output csv or txt>"
#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in kmeans_value:
        writer.writerow([val]) """
#Imported Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# Importing the f
df = pd.read_csv(r"https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv")
df = pd.DataFrame(df)
scaler = MinMaxScaler()
stdf = scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
stdf = scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

km = KMeans(n_clusters=3)
clust = km.fit_predict(df[['Age','Income($)']])
cluster_centers =km.cluster_centers_


class FunctionInfo:
	
	def __init__(self):
		
		# Function information dictionary - { function name: [ list of arguments, return type, funtion description ] }
		
		self.func_dict =   {	
								'TKPY_Clust_Centers': [['float'], ['matrix'], ['cluster centers']],
                                'TKPY_predVal' : [['float','float'],['float'],['Predict the cluster for a datapoint']],
								'TKPY_DataCluster': [['float'],['array'],['clusters in the dataset']],
								"TKPY_elbow_plot" : [['float'],['array'],['elbow plot for clustering']]

							}
	def get_func_dict(self):
		return self.func_dict




# a is a dummy value, ignore it!!

def TKPY_DataCluster(a):
	clusters = clust + a -a

	return  np.array(clusters)

def TKPY_predVal(a,b):
	pred= km.predict([[a,b]])
	return(float(pred))

def TKPY_Clust_Centers(a):
	points = cluster_centers + a-a
	return np.array(points)

def TKPY_elbow_plot(a):
	sse = []

	k_rng = range(1,10)
	for k in k_rng:
		km = KMeans(n_clusters=k)
		km.fit(df[['Age','Income($)']])
		sse.append(km.inertia_)
	sse_array = np.array(sse) + a - a
	return sse_array





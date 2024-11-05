from cProfile import label
import numpy as np
import pandas as pd
import re
import math
import umap
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from sklearn.cluster import DBSCAN

# from sklearn.svm import OneClassSVM
# from pmdarima.arima import auto_arima
# from sklearn.utils import shuffle
# from statsmodels.tsa.arima.model import ARIMA
# from pmdarima import auto_arima
# import warnings
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
# warnings.simplefilter('ignore', ConvergenceWarning)




#Input: the df with the multivariate time series data
# Output: a df with a score column calculated using Isolation Forest
def iforest(df):
    model = IsolationForest(n_estimators=1000, contamination=0.04, max_samples="auto").fit(df)

    gt = pd.DataFrame(columns=['score'])
    gt['score'] = 100*(model.decision_function(df) + 0.5)
    return gt

    
    

#Parameters: -a DF with UNIVARIATE time series data -SW: the number of lagged periods desired for the model
#Output: Returns an array with the absoulute errors between the predictions using XGBoost and the actual data for the univariate time series data

def XGBoost_3(df,SW): #For univariate TS
    #Add the shifted values to each timestamp
    for i in range(SW):
        df[f"Value t-{i+1} "] = df.iloc[:,0].shift(i+1)

    M = df.shape[0] #M observations  
    errors_array = [] #Empty errors array

    #Just use the timeperiods on the window to train
    X= df.iloc[:,1:]
    y= df.iloc[:,0]

    reg = XGBRegressor(n_estimators=1000).fit(X, y, eval_set=[(X,y)],
                                                early_stopping_rounds=50, verbose=False) 
    #Findings DataFrame
    predictions= reg.predict(X.iloc[SW-1:,:])
    errors_array.append(abs(predictions-y[SW-1:]))
    return errors_array


#Multivariate XGBoost
#Parameters: -a DF with Multivariate time series data -SW: the number of lagged periods desired for the model
#Output: Returns an array with the absoulute errors between the predictions using XGBoost and the actual data for each of the features in the df
def mtsXGBoost(df,SW):
    mts_arrays = []

    for i in df.columns:
        data = df.copy()
        array_n = XGBoost_3(data[[i]], SW)
        mts_arrays.append(array_n)
    mts_arrays = np.array(mts_arrays).transpose()           
    return pd.DataFrame(mts_arrays.reshape(mts_arrays.shape[0], mts_arrays.shape[2]))

#Parameters: dataframe(m*n) 
#output:A dimension reduced dataframe with shape(m*2), dimension reduction is achieved using UMAP algorithm 
def reductionUMAP(df):
	#默認情況下，UMAP 會縮減為 2D
	reducer = umap.UMAP(random_state=42, n_neighbors=5, min_dist=0.3, metric='correlation', n_components=2).fit(df)

	#embedding將目標數組繪製為標準散點圖和顏色（因為它適用於與原始數據順序相同的轉換數據）
	embedding = reducer.fit_transform(df.values)
 
	return pd.DataFrame(embedding)


"""### UMAP has several hyperparameters that can have a significant impact on the resulting embedding. 
`n_neighbors` :  It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.
這決定了在流形結構的局部近似中使用的相鄰點的數量。較大的值將導致在丟失詳細的局部結構的情況下保留更多的全局結構。一般來說，這個參數應該經常在 5 到 50 的範圍內，選擇 10 到 15 是一個合理的默認值。

`min_dist`:controls how tightly UMAP is allowed to pack points together. 
允許將點壓縮在一起的緊密程度。較大的值確保嵌入點分佈更均勻
合理的值在 0.001 到 0.5 的範圍內，0.1 是合理的默認值

`n_components`: allows the user to determine the dimensionality of the reduced dimension space we will be embedding the data into.


`metric`:This controls how distance is computed in the ambient space of the input data. By default UMAP supports a wide variety of metrics
測量輸入空間中距離的度量的選擇
"""


#Input: df
#Output:Labels of each of the datapoints across the time series derived using DBSCAN Clustering Algorithm
def clustering_dbscan(df):
    dbscan=DBSCAN(eps =  0.5, min_samples = 256)
    model = dbscan.fit(df)
    colors = model.labels_
    return colors

Multivariate Time Series Anomaly Detection
============
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
## Table of Contents
* [How to Run?](#how-to-run)
* [File Structure](#file-structure)
* [Function Description](#function-description)
* [Reference](#references)

## How to Run?
1. Install dependencies
```
pip install -r requirements.txt
```
2. Run app.py
```
cd src
streamlit run app.py
```

## File Structure
```
root
  ├── src
  |   ├── get_anomaly.py
  |   ├── app.py
  |   ├── streamlit_funtion.py
  |   ├── params.json
  |   ├── model.py
  |   ├── process.py
  |   ├── utils.py
  │   └── plot_function.py
  ├── data 
  |   ├── sample
  |   |     ├── MTS
  |   |     └── UTS
  |   ├── RD
  │   └── SMD
  ├── assets
  |   ├── css
  |   └── img
  ├── README.md
  └── requirements.txt
```

## Function Description
### get_anomaly.py
#### I/O Overview
get_anomalies function is to identify the anomaly area in the multivariate time series given.
INPUT:
 * raw_data: dataframe::: is the multivariate time series raw data
 * gt: dataframe::: is the ground truth of the raw data
OUTPUT: 
anomalies: list of int::: for each index, assign 1 if it is an anomaly and 0 if it is not
f1_score: float::: the accuracy of the model's anomalies output
#### Variables Overview
1. hyperparameter
hyperparameter: dictionary::: loads all of the hyperparameter used for the models stored in `params.json`
**Example:** `hyperparameter['xgboost']['sliding_window']`
2. temporal: dataframe::: the result of the dimension reduction (ie. UMAP) from the temporal model (ie. XGBoost)
3. spatial: dataframe::: the result of the spatial model (ie. iForest)
4. normalized_combine: dataframe
**Columns:**
    * tmp1: normalized result #1 of `temporal`
    * tmp2: normalized result #2 of `temporal`
    * score: normalized result of `spatial`
    * upscale: arctan(`score`)
    * sum: `tmp1` + `tmp2` + `score` + `upscale`
    * is_anomaly: ground truth anomaly label
5. th: float::: the threshold for rule-based classification
#### 

### app.py
**streamlit main funtion**
1. header
It puts down some decoration on header
    - INPUT: None
    - OUTPUT: Header

2. footer
It puts down some decoration on footer
    - INPUT: None
    - OUTPUT:  Footer

3. change_position
It will change the column feature. 
    - INPUT: None
    - OUTPUT:  change position

### plot_function.py
**Note: It is highly adviced for this file to be refactored. There are many duplicate functions in streamlit_funtion. Most algorithms are also not optimized.**

1. draw_anomaly
draw_anomaly function is to draw the straight line through each of the subplot for the anomaly and show it on streamlit.
INPUT:
fig: plt.figure::: is the plot where the function will be drawing on.
df: dataframe::: is the dataframe with the anomaly label. Please put the anomaly label in the "label".
distance: int::: is the distance between each line that it will be drawn into a single range
Color: character::: is the color of the line drawn.
Alpha: float::: the opacity (transparantcy) of the colored line.
OUTPUT: Plot in streamlit

2. plot_graph
plot_graph function is to draw a complete plot of the multivariate time series, including the anomaly and show it on streamlit.
INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
gt: dataframe::: is the ground truth of the raw data
output: dataframe::: is the output of the model with a 'label' column
position_choice: list of int::: is the index of the desired columns to be shown
width: int::: the width of the graph
height: int::: the height of the graph
OUTPUT: Plot in streamlit

3. draw_anomaly_normal
draw_anomaly_normal function is to draw the straight line through each of the subplot
INPUT:
fig: plt.figure::: is the plot where the function will be drawing on.
data: dataframe::: is the dataframe with the anomaly label. Please put the anomaly label in the "label".
distance: int::: is the distance between each line that it will be drawn into a single range
color: character::: is the color of the line drawn.
alpha: float::: the opacity (transparantcy) of the colored line.
first: int::: the starting time
last: int::: the ending time
OUTPUT:
pair: list of int tuple::: start and end of each anomaly area

4. plot_graph_normal
plot_graph_normal function is to draw a complete plot of the multivariate time series, including the anomaly.
INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
gt: dataframe::: is the ground truth of the raw data
output: dataframe::: is the output of the model with a 'label' column
start: int::: the starting time
end: int::: the ending time
OUTPUT: matplotlib graph

5. index_of_anomalies
index_of_anomalies function is to get the start and end of each anomaly area. 
INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
distance: int::: is the distance between each anomaly that it will be drawn into a single area
OUTPUT:
pair: list of int tuple::: start and end of each anomaly area

6. plot_graph_only
plot_graph_only function is to just draw the graph without the anomaly area.
INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
OUTPUT: matplotlib graph

7. report_word
report_word function is to find the area with the highest percentage of anomaly in range of 200.
INPUT:
pair: list of int tuple::: start and end of each anomaly area
len_data: int::: the length of the data
OUTPUT:
highest_anomaly: tuple of int::: the area of the highest percentage of anomaly in range of 200

### streamlit_funtion.py
**It has some plot funtions from plot_function.py. But, it does a little revise for using on streamlit.**
1. load_data
upload data 
    - INPUT: csv
    - OUTPUT: return dataframe

2. plot_graph_normal_fig
plot_graph_normal function is to draw a complete plot of the multivariate time series, including the anomaly.
But, plot_graph_normal_fig only use on streamlit. It differ from plot_graph_normal at plot_function.py because it add position_choice,width and height.

    - INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
gt: dataframe::: is the ground truth of the raw data
output: dataframe::: is the output of the model with a 'label' column
start: int::: the starting time
end: int::: the ending time
position_choice: list::: choosing the column features
width: int::: for figure width
height: int::: for figure height

    - OUTPUT:
matplotlib graph on website

3. draw_anomaly_normal
draw_anomaly_normal function is to draw the straight line through each of the subplot.
But, draw_anomaly_normal only use on streamlit. 
    - INPUT:
fig: plt.figure::: is the plot where the function will be drawing on.
data: dataframe::: is the dataframe with the anomaly label. Please put the anomaly label in the “label”.
distance: int::: is the distance between each line that it will be drawn into a single range
color: character::: is the color of the line drawn.
alpha: float::: the opacity (transparantcy) of the colored line.
first: int::: the starting time
last: int::: the ending time
    - OUTPUT: 
pair: list of int tuple::: start and end of each anomaly area


4. plot_graph_normal
plot_graph_normal function is to draw a complete plot of the multivariate time series, including the anomaly.
But, plot_graph_normal only use on streamlit. 

    - INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
gt: dataframe::: is the ground truth of the raw data
output: dataframe::: is the output of the model with a ‘label’ column
start: int::: the starting time
end: int::: the ending time
position_choice: list::: choosing the column features
width: int::: for figure width
height: int::: for figure height
    - OUTPUT: matplotlib graph on website

5. index_of_anomalies
index_of_anomalies function is to get the start and end of each anomaly area.
But, index_of_anomalies only use on streamlit. 

    - INPUT:
data: dataframe::: is the multivariate time series raw data WITHOUT index
distance: int::: is the distance between each anomaly that it will be drawn into a single area
    - OUTPUT:
pair: list of int tuple::: start and end of each anomaly area

### model.py
1. IForest
    * __Description__
        - IForest Algorithm to detect possible outliers in the data
    * __Input Variable__ 
        - Dataframe with the multivariate time series data
    * __Output__
        - DataFrame with a score column calculated using Isolation Forest 
     
2. XGBoost3
    * __Description__
        - Errors between actual data and predicted data using XGBoost lag model
    * __Input Variable__ 
         - DF with UNIVARIATE time series data
         - SW: the number of lagged periods desired for the model
    * __Output__
        -  Returns an array with the absoulute errors between the predictions using XGBoost and the actual data for the   univariate time series data
    
3. MTSXGBoost
    * __Description__
        - XGBoost_3 function adaptation for Multivariate Time Series Data
    * __Input Variable__ 
         - DF with Multivariate time series data
         - SW: the number of lagged periods desired for the model
    * __Output__
        -   Returns an array with the absoulute errors between the predictions using XGBoost and the actual data for each of the features in the df
4. ReductionUMAP
    * __Description__
        - Dimension Reduction using UMAP Algorithms
       
        
    * __Input Variable__ 
         - dataframe with dimensions (m*n) 
    * __Output__
        -  A dimension reduced dataframe with shape(m*2), dimension reduction is achieved using UMAP algorithm 


### process.py
1. merge_all_csv
    * __Description__
        - load all csv and merge with time
    * __Input Variable__ 
        - path : (str) where the file place
    * __Variable__
        - num :  point to the column name
        - because every csv only got time and unique column name
        - all_file_name : list of all folder in that directory
        - column_nmae : the name to be set for every second column in csv
        - content : first read csv to be merge later
        - new_content : get the file data ,name is given for every column to merge success
    * __Suggestion To Improve__
        - os.chdir should be outside 

2. change_gt
    * __Description__
        - change ground truth according model output within accepted range
    * __Input Variable__ 
        - gt: (np.array) input must be 0,1
        - mod_out : (np.array) input must be 0,1 and same length with gt
        - interval: (int) distance range accepted for anomaly point
    * __Variable__
        - am : get np.maskarray which does not have 0
        - slices : get list of start end of 1 by mode_out


### utils.py
ref. https://cs230.stanford.edu/blog/hyperparameters/
將json 引入
定義一函數名稱 撰寫內容來自第一行ref.
設定一參數f 使其以唯獨方式'r' 打開一個叫做'params.json' 的文件
設定一參數data 使其讀取'f' 的json檔案
關閉已開啟的這個'f' 檔案
回傳讀取的'data' 內容

## References
[OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)
目前 RDteam 使用的 public dataset 接下來測試跟設計也要把這份資料一起加入討論

https://github.com/JanpuHou/stock_visulization_app

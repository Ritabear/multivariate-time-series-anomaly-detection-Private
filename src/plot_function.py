from xmlrpc.client import MAXINT
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import itertools

# draw_anomaly function is to draw the straight line through each of the subplot for the anomaly and show it on streamlit.
# INPUT:
# fig: plt.figure::: is the plot where the function will be drawing on.
# df: dataframe::: is the dataframe with the anomaly label. Please put the anomaly label in the "label".
# distance: int::: is the distance between each line that it will be drawn into a single range
# Color: character::: is the color of the line drawn.
# Alpha: float::: the opacity (transparantcy) of the colored line.
# OUTPUT: Plot in streamlit
def draw_anomaly(fig, df, distance, color, alpha):
    # gt_anom = df[df["label"] == 1] #all the index of anomalies.
    # 上傳的csv 欄位重要，不然會取錯
    gt_anom = df[df[0] == 1]

    # Initial declaration for variables
    # x0: int::: is the index in front of the list
    # x1: int::: is the index behind. (AKA x0 < x1)
    # range: bool::: to indicate if it is currently drawing a single range
    # start: int or date::: the start of the of the range
    # end: int or date::: the end of the range
    x0 = -1
    x1 = -1
    line = 0
    start = MAXINT
    end = -1

    for i in gt_anom.index:
        if x0 == -1:
            # assign the first value for the
            x0 = i
        else:
            x1 = x0
            x0 = i
            if abs(x0-x1) < distance:
                line = 1
                start = min(start, x0)
                end = max(end, x1)
            elif line == 1:
                [j.axvspan(start-20, end+20, facecolor=color, alpha=alpha)
                 for j in fig.get_axes()]
                line = 0
                start = MAXINT
                end = -1
            else:
                [j.axvspan(x1-50, x1+50, facecolor=color, alpha=alpha)
                 for j in fig.get_axes()]

# plot_graph function is to draw a complete plot of the multivariate time series, including the anomaly and show it on streamlit.
# INPUT:
# data: dataframe::: is the multivariate time series raw data WITHOUT index
# gt: dataframe::: is the ground truth of the raw data
# output: dataframe::: is the output of the model with a 'label' column
# position_choice: list of int::: is the index of the desired columns to be shown
# width: int::: the width of the graph
# height: int::: the height of the graph
# OUTPUT: Plot in streamlit
def plot_graph(data, gt, output, position_choice, width, height):
    df = data.copy()
    df_output = output.copy()
    df = df.reset_index(inplace=False)

    # col = df.columns.to_list()
    fig = plt.figure(figsize=(width, height), dpi=72)

    ax = []
    for i in range(1, len(position_choice)):
        # Make subplots with sharex
        if i == 1:
            ax.append(plt.subplot(len(position_choice), 1, i))
        else:
            ax.append(plt.subplot(len(position_choice), 1, i, sharex=ax[0]))

        plt.plot(df['index'], df[position_choice[i]], color="#542de0")
        plt.title(position_choice[i], fontsize=12, loc="right")  # Subplot Name

        # declare grid and place behind everything else
        ax[i-1].set_axisbelow(True)
        ax[i-1].grid(True, color="#ffffff", linewidth=2)

        # Set y-limit
        min_range = min(df[position_choice[i]])-10
        max_range = max(df[position_choice[i]])+10
        plt.ylim(min_range, max_range)

        # Aestheticc: Hide outside border and set background color
        ax[i-1].spines['top'].set_visible(False)
        ax[i-1].spines['bottom'].set_visible(False)
        ax[i-1].spines['left'].set_visible(False)
        ax[i-1].spines['right'].set_visible(False)
        ax[i-1].set_facecolor("#ebe9f5")

    # draw_anomaly(fig, gt, 500, 'r', 0.5)
    # draw_anomaly(fig, df_output, 500, 'g', 0.7)

    st.pyplot(fig)

##########################################################################################################################
# NORMAL DRAWING
##########################################################################################################################

# draw_anomaly_normal function is to draw the straight line through each of the subplot
# INPUT:
# fig: plt.figure::: is the plot where the function will be drawing on.
# data: dataframe::: is the dataframe with the anomaly label. Please put the anomaly label in the "label".
# distance: int::: is the distance between each line that it will be drawn into a single range
# color: character::: is the color of the line drawn.
# alpha: float::: the opacity (transparantcy) of the colored line.
# first: int::: the starting time
# last: int::: the ending time
# OUTPUT:
# pair: list of int tuple::: start and end of each anomaly area
def draw_anomaly_normal(fig, data, distance, color, alpha, first, last):
    pair = index_of_anomalies(data, 5)
    # i and j ,start and stop for ploting
    for start, end in pair:
        # 小於開始值，未大於範圍
        if start < first and end < last:
            [j.axvspan(first, end+20, facecolor=color, alpha=alpha)
             for j in fig.get_axes()]
        if start > first and end > last:
            [j.axvspan(start-20, end, facecolor=color, alpha=alpha)
             for j in fig.get_axes()]
        else:
            [j.axvspan(start-50, end+50, facecolor=color, alpha=alpha)
             for j in fig.get_axes()]
    return pair

# plot_graph_normal function is to draw a complete plot of the multivariate time series, including the anomaly.
# INPUT:
# data: dataframe::: is the multivariate time series raw data WITHOUT index
# gt: dataframe::: is the ground truth of the raw data
# output: dataframe::: is the output of the model with a 'label' column
# start: int::: the starting time
# end: int::: the ending time
# OUTPUT: matplotlib graph
def plot_graph_normal(data, gt, output, start, end):
    df = data.copy()
    df = df[start:end]
    df = df.reset_index(inplace=False)

    gt = gt[start:end]
    output = output[start:end]

    col = df.columns.to_list()
    fig = plt.figure(figsize=(14, 84), dpi=72)

    ax = []
    for i in range(1, len(col)):
        # Make subplots with sharex
        if i == 1:
            ax.append(plt.subplot(len(col), 1, i))
        else:
            ax.append(plt.subplot(len(col), 1, i, sharex=ax[0]))

        plt.plot(df['index'], df[col[i]], color="#542de0")
        plt.title(col[i], fontsize=12, loc="right")  # Subplot Name

        # declare grid and place behind everything else
        ax[i-1].set_axisbelow(True)
        ax[i-1].grid(True, color="#ffffff", linewidth=2)

        # Set y-limit
        min_range = min(df[col[i]])
        max_range = max(df[col[i]])
        plt.ylim(min_range, max_range)

        # Aestheticc: Hide outside border and set background color
        ax[i-1].spines['top'].set_visible(False)
        ax[i-1].spines['bottom'].set_visible(False)
        ax[i-1].spines['left'].set_visible(False)
        ax[i-1].spines['right'].set_visible(False)
        ax[i-1].set_facecolor("#ebe9f5")

    #gt_anom_1 = draw_anomaly_normal(fig, gt, 500, 'r', 0.5,start,end)
    out_anom_2 = draw_anomaly_normal(fig, output, 500, 'r', 0.7, start, end)

    plt.tight_layout()
    plt.show()
    return out_anom_2

# index_of_anomalies function is to get the start and end of each anomaly area. 
# INPUT:
# data: dataframe::: is the multivariate time series raw data WITHOUT index
# distance: int::: is the distance between each anomaly that it will be drawn into a single area
# OUTPUT:
# pair: list of int tuple::: start and end of each anomaly area
def index_of_anomalies(data, distance):
   # print("data", data)
   # gt_anom = data[data[0] == 1]
    gt_anom = data[data["label"] == 1]
    current = gt_anom.index[0]  # ex.175
    pair = []
    for i in range(1, len(gt_anom)):
        if abs(gt_anom.index[i-1] - gt_anom.index[i]) <= distance:
            continue
        else:
            pair.append((current, gt_anom.index[i-1]))
            current = gt_anom.index[i]
    return pair

# plot_graph_only function is to just draw the graph without the anomaly area.
# INPUT:
# data: dataframe::: is the multivariate time series raw data WITHOUT index
# OUTPUT: matplotlib graph
def plot_graph_only(data):
    df = data.copy()
    df = df.reset_index(inplace=False)

    col = df.columns.to_list()
    fig = plt.figure(figsize=(14, 14), dpi=72)

    ax = []
    for i in range(1, len(col)):
        # Make subplots with sharex
        if i == 1:
            ax.append(plt.subplot(len(col), 1, i))
        else:
            ax.append(plt.subplot(len(col), 1, i, sharex=ax[0]))

        plt.plot(df['index'], df[col[i]], color="#542de0")
        plt.title(col[i], fontsize=12, loc="right")  # Subplot Name

        # declare grid and place behind everything else
        ax[i-1].set_axisbelow(True)
        ax[i-1].grid(True, color="#ffffff", linewidth=2)

        # Set y-limit
        # min_range = min(df[col[i]])-10
        # max_range = max(df[col[i]])+10
        # plt.ylim(min_range, max_range)

        # Aestheticc: Hide outside border and set background color
        ax[i-1].spines['top'].set_visible(False)
        ax[i-1].spines['bottom'].set_visible(False)
        ax[i-1].spines['left'].set_visible(False)
        ax[i-1].spines['right'].set_visible(False)
        ax[i-1].set_facecolor("#ebe9f5")

    plt.tight_layout()
    plt.show()

# report_word function is to find the area with the highest percentage of anomaly in range of 200.
# INPUT:
# pair: list of int tuple::: start and end of each anomaly area
# len_data: int::: the length of the data
# OUTPUT:
# highest_anomaly: tuple of int::: the area of the highest percentage of anomaly in range of 200
def report_word(pair, len_data):
    start = 0
    highest_anomaly = (0,0) # to see the range which has high anomaly
    previous = 0
    x = 0
    for num in range(200,len_data,200):
        end = num 
        res = list(filter(lambda sub : all(ele >= start and ele <= end for ele in sub), pair))
        # get tuple [0] and [1] to different list
        if res != None:
            elem_1 = [res[i][0] for i in range(len(res))]
            elem_2 = [res[i][1] for i in range(len(res))]
            after_sub = list(map(lambda i, j: i - j, elem_2, elem_1))
        
            for i in after_sub:
                if i == 0:
                    x+=1
                else:
                    x+=i
            
            if x > previous:
                highest_anomaly = (elem_1[0] , elem_2[-1])
                previous = x
                x = 0
                start = end
            
            else:
                x = 0
                start = end
 
        # last
        if len_data - end < 200:
            end = len_data
            res = list(filter(lambda sub : all(ele >= start and ele <= end for ele in sub), pair))
            # get tuple [0] and [1] to different list
            if res != None:
                elem_1 = [res[i][0] for i in range(len(res))]
                elem_2 = [res[i][1] for i in range(len(res))]
                after_sub = list(map(lambda i, j: i - j, elem_2, elem_1))
            
                for i in after_sub:
                    if i == 0:
                        x+=1
                    else:
                        x+=i
                
                if x > previous:
                    highest_anomaly = (elem_1[0] , elem_2[-1])
                    previous = x
                    x = 0
                    start = end
                
                else:
                    x = 0
                    start = end

    return highest_anomaly

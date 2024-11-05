from PIL import Image
from xmlrpc.client import MAXINT
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


@st.cache(allow_output_mutation=True, ttl=60 * 10, max_entries=20)
def load_data(path):
    df = pd.read_csv(path, header=None)
    return df


def plot_graph_normal_fig(data, gt, output, position_choice, start, end, width, height):
    df = data.copy()
    # print("plot_graph_normal")
    # print(df)
    df = df[start:end]
    df = df.reset_index(inplace=False)

    # print("df-------")
    # print(df)

    gt = gt[start:end]
    output = output[start:end]
    # print("gt-------")
    # print(gt)
    # print("output-------")
    # print(output)

    # col = df.columns.to_list()
    col = position_choice
    # print("*****")
    # print(col)
    fig = plt.figure(figsize=(width, height), dpi=72)

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
    # print("output")
    # print(output)
    #gt_anom_1 = draw_anomaly_normal(fig, gt, 500, 'r', 0.5,start,end)
    out_anom_2 = draw_anomaly_normal(fig, output, 10, 'r', 0.7, start, end)

    # plt.tight_layout()
    # plt.show()
    #
    st.pyplot(fig)
    # return out_anom_2


##########################################################################################################################
# NORMAL DRAWING
##########################################################################################################################
def draw_anomaly_normal(fig, data, distance, color, alpha, first, last):
    # print("data:")
    # print(data)
    pair = index_of_anomalies(data, 5)
    # i and j ,start and stop for ploting
    for start, end in pair:
        # 小於開始值，未大於範圍
        if start < first and end < last:
            [j.axvspan(first, end, facecolor=color, alpha=alpha)
             for j in fig.get_axes()]
        if start > first and end > last:
            [j.axvspan(start, end, facecolor=color, alpha=alpha)
             for j in fig.get_axes()]
        else:
            [j.axvspan(start, end, facecolor=color, alpha=alpha)
             for j in fig.get_axes()]
    return pair


# plot_graph function is to draw a complete plot of an anomaly.
# data: dataframe::: is the spatial raw data WITHOUT index
# gt: dataframe::: is the ground truth of the raw data
# output: dataframe::: is the output of the model with a 'label' column

def plot_graph_normal(data, gt, output, position_choice, start, end, width, height):
    df = data.copy()
    # print("plot_graph_normal")
    # print(df)
    df = df[start:end]
    df = df.reset_index(inplace=False)

    # print("df-------")
    # print(df)

    gt = gt[start:end]
    output = output[start:end]
    # print("gt-------")
    # print(gt)
    # print("output-------")
    # print(output)

    # col = df.columns.to_list()
    col = position_choice
    print("*****")
    print(col)
    fig = plt.figure(figsize=(width, height), dpi=72)

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
    # print("output")
    # print(output)
    #gt_anom_1 = draw_anomaly_normal(fig, gt, 500, 'r', 0.5,start,end)
    out_anom_2 = draw_anomaly_normal(fig, output, 500, 'r', 0.7, start, end)

    # plt.tight_layout()
    # plt.show()
    #
    # st.pyplot(fig)
    return out_anom_2


def index_of_anomalies(data, distance):
    # gt_anom = data[data["label"] == 1]
    # print("index_of_anomalies")
    # print(data)
    gt_anom = data[data[0] == 1]
    current = gt_anom.index[0]  # ex.175
    pair = []
    for i in range(1, len(gt_anom)):
        if abs(gt_anom.index[i-1] - gt_anom.index[i]) <= distance:
            continue
        else:
            pair.append((current, gt_anom.index[i-1]))
            current = gt_anom.index[i]
    return pair

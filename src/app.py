import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
from xmlrpc.client import MAXINT
from PIL import Image
import pandas as pd

from get_anomaly import get_anomalies
from streamlit_funtion import load_data, plot_graph_normal, plot_graph_normal_fig
import plot_function

def header():
    st.markdown('<style>' + open('../assets/css/style.css').read() +
                '</style>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    col3.image("../assets/img/icon_400.png", width=150)
    st.markdown("<h1 style='text-align: center; color: white;'>Cloud Resource Usage Health Check</h1>",
                unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: gray;font-size:10px; margin-right:100px;'>Caloudi &nbsp;&nbsp; Version:1.0.0 &nbsp;&nbsp; Contact:zxc@gmail.com</div>", unsafe_allow_html=True)

    # footer
    # hide made in streamlit
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def footer():
    st.markdown("<div style='height:300px;'> </div>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white; margin-right:150px;'>Professional Service</h3>",
                unsafe_allow_html=True)

    image_service = Image.open("../assets/img/service.png")
    new_service = image_service.resize((1800, 900))
    st.image(new_service, use_column_width=True)

    st.markdown("<h3 style='text-align: center; color: white; margin-right:150px;'>Model Pipeline</h3>",
                unsafe_allow_html=True)

    image = Image.open("../assets/img/model.png")
    new_image = image.resize((1000, 350))
    st.image(new_image, use_column_width=True)

def change_position():
    header()
    uploaded_file1 = st.file_uploader("Choose a Raw File:",
                                    type=['csv'],
                                    help="only upload one csv file",
                                    key = "upload_file",
                                    accept_multiple_files=False)

    st.sidebar.markdown('### Parameter setting')
    st.sidebar.slider("plot width", min_value=15, max_value=20, step=5, key="width")
    st.sidebar.slider("plot height", min_value=25, max_value=100, step=1, key="height")

    st.sidebar.multiselect(
        'Choose column:',
        options=st.session_state.raw_data.columns.values.tolist(),
        default=st.session_state['position'],
        on_change = change_position,
        key = 'position'
    )

    st.markdown("<h3 style='text-align: center; color: white;'>Multivariate Time Series Anomaly Detection</h3>",
                unsafe_allow_html=True)
    plot_graph_normal_fig(st.session_state.raw_data,
                            st.session_state.gt,
                            st.session_state.anomaly,
                            st.session_state['position'], 0,
                            st.session_state.raw_data.shape[0],
                            st.session_state['width'],
                            st.session_state['height'])
    st.write(st.session_state.report_text)
    footer()


st.set_page_config(page_title="Caloudi: Cloud Health check", layout="wide", initial_sidebar_state="collapsed", page_icon="🧊",
                    menu_items={"About": "This is an awesome App for detecting amonoly !"})

header()
st.session_state.uploaded = False
uploaded_file1 = st.file_uploader("Choose a Raw File:",
                                    type=['csv'],
                                    help="only upload one csv file",
                                    key = "upload_file",
                                    accept_multiple_files=False)

btn1, btn2, btn3, btn4, btn5 = st.columns([1, 1, 1, 1, 1])
if btn3.button('🔍Calculate'):
    if uploaded_file1 is not None:
        st.session_state.raw_data = load_data(uploaded_file1)
        gt_path = "../data/SMD/test_label/"+uploaded_file1.name
        # print("path: ", gt_path)
        st.session_state.gt = load_data(gt_path)
        # print("load data finished\n")

        st.balloons()
        st.markdown("<div style='height:100px;'> </div>", unsafe_allow_html=True)

        with st.spinner(text='In progress...'):
            st.session_state.anomaly, st.session_state.f1 = get_anomalies(raw_data = st.session_state.raw_data, 
                                                                            gt = st.session_state.gt)
                    
        st.session_state.uploaded = True

if st.session_state.uploaded:
    st.sidebar.markdown('### Parameter setting')
    width = st.sidebar.slider("plot width", min_value=15, max_value=20, step=5, key="width")
    height = st.sidebar.slider("plot height", min_value=25, max_value=100, step=1, key="height")

    column = st.session_state.raw_data.columns.values.tolist()
    # print(column)
    position_choice = st.sidebar.multiselect(
        'Choose column:',
        options=column,
        default=column,
        on_change = change_position,
        key = 'position'
    )

    st.markdown("<h3 style='text-align: center; color: white;'>Multivariate Time Series Anomaly Detection</h3>",
                unsafe_allow_html=True)

    plot_graph_normal_fig(st.session_state.raw_data,
                            st.session_state.gt,
                            st.session_state.anomaly,
                            st.session_state['position'], 0,
                            st.session_state.raw_data.shape[0],
                            st.session_state['width'],
                            st.session_state['height'])

    pair = plot_graph_normal(st.session_state.raw_data,
                            st.session_state.gt,
                            st.session_state.anomaly,
                            st.session_state['position'], 0,
                            st.session_state.raw_data.shape[0],
                            st.session_state['width'],
                            st.session_state['height'])
    highest_anomaly = plot_function.report_word(pair, len(st.session_state.gt))
    st.session_state.report_text = f"From timestep {highest_anomaly[0]} to {highest_anomaly[1]} is the interval which has the most anomaly point."
    st.write(st.session_state.report_text)

    footer()
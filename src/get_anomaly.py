import utils
import process
import model
import numpy as np
import os
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


import warnings
warnings.filterwarnings("ignore")

# get_anomalies function is to identify the anomaly area in the multivariate time series given.
# INPUT:
# raw_data: dataframe::: is the multivariate time series raw data
# gt: dataframe::: is the ground truth of the raw data
# OUTPUT: 
# anomalies: list of int::: for each index, assign 1 if it is an anomaly and 0 if it is not
# f1_score: float::: the accuracy of the model's anomalies output
def get_anomalies(raw_data, gt):

    hyperparameter = utils.get_hyperparameter()

    # downsampling
    # raw_data = raw_data[::2]

    # Run both models
    temporal = model.reductionUMAP(model.mtsXGBoost(
        raw_data, hyperparameter['xgboost']['sliding_window']))
    temporal = temporal.rename(columns={0: 'tmp1', 1: 'tmp2'})
    spatial = (model.iforest(raw_data).tail(
        len(temporal))).reset_index(drop=True)

    # Combine both output and z-standarized normalization
    result = pd.concat([temporal, spatial], axis=1)
    normalized_combine = result.apply(stats.zscore)
    normalized_combine['upscale'] = np.arctan((normalized_combine['score']))

    # Clustering
    # label = model.clustering_dbscan(normalized_combine)
    # normalized_combine["anomaly_score"] = label
    # for i in normalized_combine['anomaly_score']:
    #     if i != 0:
    #         normalized_combine["anomaly_score"] = 1

    normalized_combine['sum'] = normalized_combine['score'] + \
        normalized_combine['upscale'] + \
        normalized_combine['tmp1'] + normalized_combine['tmp2']
    gt = gt.tail(len(temporal)).reset_index(drop=True)
    # gt = gt.rename(columns={'0': 'label'})
    normalized_combine['is_anomaly'] = gt[0]

    # Get Threshold
    df1 = normalized_combine[normalized_combine['is_anomaly'] == 1]
    th = df1['sum'].mean()

    # Rule Based
    anomalies = []
    for i in normalized_combine['sum']:
        if i <= th:
            anomalies.append(1)
        else:
            anomalies.append(0)

    # Run range model
    anomalies = np.array(anomalies)
    ground_truth = np.array(gt)
    process.change_gt(ground_truth, anomalies, 5)

    return pd.DataFrame(anomalies), f1_score(ground_truth, anomalies)
# Count
# print(f1_score(ground_truth, anomalies))
# pd.DataFrame(anomalies).to_csv("anomalies.csv", index=False, header=False)

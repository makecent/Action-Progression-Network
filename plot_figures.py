#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : plot_figures.py
# Author: Chongkai LU
# Date  : 14/7/2020
import json
import numpy as np
import pandas as pd
from .utils import *
from matplotlib import pyplot as plt


# Plot ap curves versus IoU
with open("saved/task1_imagenet_ap", 'r') as f:
    task1_ap = json.load(f)
with open("saved/task2_imagenet_ap", 'r') as f:
    task2_ap = json.load(f)
with open("saved/rim_task2.csv", 'r') as f:
    rim = np.loadtxt(f, delimiter=',')

plt.figure()
plt.plot(np.array(list(task1_ap.keys())).astype(np.float), [i*100 for i in list(task1_ap.values())], 'r-', label='Action1')
plt.plot(np.array(list(task1_ap.keys())).astype(np.float), [j*100 for j in list(task2_ap.values())], 'b--', label='Action2')
plt.plot(np.array(list(task1_ap.keys())).astype(np.float), rim*100, 'g:', label='Action2 by [29]')
plt.axis([0, 0.9, 0, 105])
plt.grid(True)
plt.legend()
plt.xlabel('IoU thresholds')
plt.ylabel('Average Precision (%)')
plt.title('AP@IoU')
plt.show()

# Plot down-sampling figure
with open("task1_imagenet_pre", 'r') as f:
    list_pre = json.load(f)
predictions = {k: np.array(v) for k, v in list_pre.items()}
ta = pd.read_csv("/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/task1.csv")
video_names = ta.iloc[:, 0].unique()
ap = {}
downsampling = [1, 2, 4, 8, 16, 32, 64, 128, 256]
for ds in downsampling:
    ground_truth = {}
    action_detected = {}
    for v, prediction in predictions.items():
        prediction = prediction[::ds]
        gt = ta.loc[ta.iloc[:, 0] == int(v)].iloc[:, 1:].values
        gt = gt // ds
        v = str(v)
        ground_truth[v] = gt
        ads = action_search(prediction, min_T=90, max_T=10, min_L=500//ds)
        action_detected[v] = ads

    num_gt = sum([len(gt) for gt in ground_truth.values()])
    loss = np.vstack(list(action_detected.values()))[:, 2]
    d_ap = {}

    for IoU in range(1, 91):
        IoU *= 0.01
        IoU_tps = {}
        for v in video_names:
            v = str(v)
            IoU_tps[v] = calc_truepositive(action_detected[v], ground_truth[v], IoU)

        IoU_tp_values = np.hstack(list(IoU_tps.values()))
        IoU_ap = average_precision(IoU_tp_values, num_gt, loss)
        d_ap[IoU] = IoU_ap
    ap[ds] = d_ap

plt.figure()
for k, v in ap.items():
    plt.plot(np.array(list(v.keys())).astype(np.float), [i*100 for i in list(v.values())], label=k)
    plt.axis([0, 0.9, 0, 105])
    plt.grid(True)
    plt.legend()
    plt.xlabel('IoU thresholds')
    plt.ylabel('Average Precision (%)')
    plt.title('AP@IoU at different down-sampling rate')
plt.show()


# Plot orthogonality figure
from scipy.signal import savgol_filter
with open("task1_imagenet_pre", 'r') as f:
    task1_list_pre = json.load(f)
with open("task2_imagenet_pre", 'r') as f:
    task2_list_pre = json.load(f)

task1_predictions = {k: np.array(v) for k, v in task1_list_pre.items()}
task2_predictions = {k: np.array(v) for k, v in task2_list_pre.items()}

ground_truth = {}
action_detected = {}
v, task1_prediction = list(task1_predictions.items())[0]
task2_prediction = task2_predictions[v]

task1_ta = pd.read_csv("/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/task1.csv", header=None)
task2_ta = pd.read_csv("/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/task2.csv", header=None)
task1_gt = task1_ta.loc[task1_ta.iloc[:, 0] == int(v)].iloc[:, 1:].values
task2_gt = task2_ta.loc[task2_ta.iloc[:, 0] == int(v)].iloc[:, 1:].values
v = str(v)
task1_ads = action_search(task1_prediction, min_T=90, max_T=10, min_L=500)
task2_ads = action_search(task2_prediction, min_T=75, max_T=10, min_L=500)

plt.figure()
plt.plot(savgol_filter(task1_prediction, 501, 3), 'k-', label='Action1 pre')
plt.plot(savgol_filter(task2_prediction, 501, 3), 'b-', label='Action2 pre')
plt.vlines(task1_gt[:, 0], 0, 90, colors='r', linestyles='dashed', label='Action1 gt')
plt.vlines(task1_gt[:, 1], 0, 90, colors='r', linestyles='dashed')
plt.vlines(task2_gt[:, 0], 0, 90, colors='g', linestyles='dashed', label='Action2 gt')
plt.vlines(task2_gt[:, 1], 0, 90, colors='g', linestyles='dashed')
plt.yticks(np.arange(0, 100, 20.0))
plt.xlabel('Frame Index')
plt.ylabel('Action Progressions')
plt.grid()
plt.legend()
plt.title("Orthogonality among detections for different actions")
plt.show()
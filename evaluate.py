#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : evaluate.py
# Author: Chongkai LU
# Date  : 14/7/2020
import numpy as np
import pandas as pd
import json
from utils import *

# Settings
setting = "scratch"  # or "imagenet" or "scratch_multi-task" or "imagenet_multi-task"
# %% Load predictions
with open("saved_predictions/task1_{}_pre".format(setting), 'r') as f:
    list_task1_pre = json.load(f)
with open("saved_predictions/task2_{}_pre".format(setting), 'r') as f:
    list_task2_pre = json.load(f)

task1_predictions = {k: np.array(v) for k, v in list_task1_pre.items()}
task2_predictions = {k: np.array(v) for k, v in list_task2_pre.items()}
video_list = list(task1_predictions.keys())   # all videos in DFMAD-70 contain both task1 and task2
# %% Load ground truth temporal annotations
anndir = "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test"
task1_ta = pd.read_csv(anndir + '/task1.csv')
task2_ta = pd.read_csv(anndir + '/task2.csv')

# %% Complete Action Search
task1_ground_truth = {}   # Similar with "task1_ta" while this is in form of python dictionary and keys are video names.
task2_ground_truth = {}
task1_action_detected = {}
task2_action_detected = {}

for v, prediction in task1_predictions.items():
    gt = task1_ta.loc[task1_ta.iloc[:, 0] == int(v)].iloc[:, 1:].values
    v = str(v)
    task1_ground_truth[v] = gt
    ads = action_search(prediction, min_T=90, max_T=10, min_L=500)
    task1_action_detected[v] = ads

for v, prediction in task2_predictions.items():
    gt = task2_ta.loc[task2_ta.iloc[:, 0] == int(v)].iloc[:, 1:].values
    v = str(v)
    task2_ground_truth[v] = gt
    ads = action_search(prediction, min_T=75, max_T=10, min_L=500)
    task2_action_detected[v] = ads

# %% Calculate average precision
task1_num_gt = sum([len(gt) for gt in task1_ground_truth.values()])
task2_num_gt = sum([len(gt) for gt in task2_ground_truth.values()])
task1_loss = np.vstack(list(task1_action_detected.values()))[:, 2]
task2_loss = np.vstack(list(task2_action_detected.values()))[:, 2]

task1_ap = {}
task2_ap = {}
for IoU in range(1, 91):
    IoU *= 0.01
    task1_IoU_tps = {}
    task2_IoU_tps = {}
    for v in video_list:
        v = str(v)
        task1_IoU_tps[v] = calc_truepositive(task1_action_detected[v], task1_ground_truth[v], IoU)
        task2_IoU_tps[v] = calc_truepositive(task2_action_detected[v], task2_ground_truth[v], IoU)

    task1_IoU_tp_values = np.hstack(list(task1_IoU_tps.values()))
    task2_IoU_tp_values = np.hstack(list(task2_IoU_tps.values()))
    task1_IoU_ap = average_precision(task1_IoU_tp_values, task1_num_gt, task1_loss)
    task2_IoU_ap = average_precision(task2_IoU_tp_values, task2_num_gt, task2_loss)
    task1_ap["{:.2f}".format(IoU)] = task1_IoU_ap
    task2_ap["{:.2f}".format(IoU)] = task2_IoU_ap

# %% Save average precisions
with open("task1_{}_ap".format(setting), 'w') as f:
    json.dump(task1_ap, f)

with open("task2_{}_ap".format(setting), 'w') as f:
    json.dump(task2_ap, f)

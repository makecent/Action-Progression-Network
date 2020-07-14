#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : evaluation.py
# Author: Chongkai LU
# Date  : 12/7/2020
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from pathlib import Path
from load_data import *
from utils import *

pretrain = 'scratch'
action = "task2"
model_path = "/mnt/louis-consistent/Saved/DFMAD-70_output/task2/Model/2020-07-13-23-51-59/07-11.23.h5"

annfile = "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/{}.csv".format(action)
temporal_annotation = pd.read_csv(annfile, header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
test_dir = "/mnt/louis-consistent/Datasets/DFMAD-70/Images/test"
n_mae = normalize_mae(101)  # make mae loss normalized into range 0 - 100.

# %% Prediction
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', metrics=[n_mae])

predictions = {}
ground_truth = {}
for v in video_names:

    video_path = Path(test_dir, str(v))
    img_list = find_imgs(video_path)
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)

    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.squeeze(prediction)

with open('{}_{}_pre'.format(action, pretrain), 'w') as f:
    list_pre = {str(k): v.tolist() for k, v in predictions.items()}
    json.dump(list_pre, f)


# %% Detect actions
with open("{}_{}_pre".format(action, pretrain), 'r') as f:
    list_pre = json.load(f)

predictions = {k: np.array(v) for k, v in list_pre.items()}
ground_truth = {}
action_detected = {}
for v, prediction in predictions.items():
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == int(v)].iloc[:, 1:].values
    v = str(v)
    ground_truth[v] = gt
    ads = action_search(prediction, min_T=75 if action == 'task2' else 90, max_T=10, min_L=500)
    action_detected[v] = ads

num_gt = sum([len(gt) for gt in ground_truth.values()])
loss = np.vstack(list(action_detected.values()))[:, 2]
ap = {}
for IoU in range(1, 91):
    IoU *= 0.01
    IoU_tps = {}
    for v in video_names:
        v = str(v)
        IoU_tps[v] = calc_truepositive(action_detected[v], ground_truth[v], IoU)

    IoU_tp_values = np.hstack(list(IoU_tps.values()))
    IoU_ap = average_precision(IoU_tp_values, num_gt, loss)
    ap[IoU] = IoU_ap

with open("{}_{}_ap".format(action, pretrain), 'w') as f:
    json.dump(ap, f)


# v = str(video_names[0])
# plot_detection(predictions[v], ground_truth[v], action_detected[v])


# with open("task1_imagenet_ap", 'r') as f:
#     task1_ap = json.load(f)
# with open("task2_imagenet_ap", 'r') as f:
#     task2_ap = json.load(f)
#
# plt.figure()
# plt.plot(np.array(list(task1_ap.keys())).astype(np.float), list(task1_ap.values()), 'r-', label='Action 1')
# # plt.plot(np.array(list(task1_ap.keys())).astype(np.float), list(task2_ap.values()), 'r--')
# plt.plot(np.array(list(task1_ap.keys())).astype(np.float), rim, 'g-.', label='Action 2 by [23]')
# plt.axis([0, 0.9, 0, 1.05])
# plt.grid(True)
# plt.legend()
# plt.show()
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : predict.py
# Author: Chongkai LU
# Date  : 16/7/2020

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt
from .load_data import *
from .utils import *

# Define names and paths
pretrain = 'scratch'  # or 'imagenet'
actoin = 'task1'  # or 'task2'

model_path = "/mnt/louis-consistent/Saved/DFMAD-70_output/{}/Model/2020-07-14-19-27-08/07-9.48.h5".format(actoin)

img_dir = "/mnt/louis-consistent/Datasets/DFMAD-70/Images/test"
annfile = "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/{}.csv".format(actoin)

# %% Load trained model
n_mae = normalize_mae(101)  # normalize loss into range 0-100
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='mse', metrics=[n_mae])

# %% Evaluation on test trimmed video clips

test_datalist = read_from_annfile(img_dir, annfile, y_range=(0, 100), mode='rgb', ordinal=False, weighted=False, stack_length=1)

test_dataset = build_dataset_from_slices(*test_datalist, batch_size=1, shuffle=False)

evaluation = model.evaluate(test_dataset)

# %% Prediction on test untrimmed videos
video_list = [vp.stem for vp in Path(img_dir).iterdir()]
predictions = {}
ground_truth = {}
for v in video_list:
    img_list = find_imgs(Path(img_dir, v))
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.squeeze(prediction)

# # Plot predictions
# v = video_list[3]
# plt.figure()
# plt.plot(predictions[v][:, 0], 'k-', label='{}'.format(v))
# plt.legend()
# plt.show()

# %% Save predictions
with open('saved/{]_{}_pre'.format(actoin, pretrain), 'w') as f:
    list_pre = {str(k): v.tolist() for k, v in predictions.items()}
    json.dump(list_pre, f)

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
actoin = 'task1'  # or 'task2', 'task3'
mode = 'rgb'  # depends on the model you have trained
stack_length = 10  # depends on the model you have trained

model_path = "/mnt/louis-consistent/Saved/DFMAD-70_output/{}/Model/2020-07-14-19-27-08/07-9.48.h5".format(actoin)

img_dir = "/mnt/louis-consistent/Datasets/DFMAD-70/Images/test"
annfile = "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/{}.csv".format(actoin)

# %% Load trained model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss='binary_crossentropy', metrics=[mae_od])

# %% Evaluation on trimmed videos (test set)
parse_function = parse_builder(i3d=True, mode=mode)
# test_datalist = read_from_annfile(img_dir, annfile, y_range=(0, 100), mode='rgb', ordinal=False, weighted=False, stack_length=1)
#
# test_dataset = build_dataset_from_slices(*test_datalist, batch_size=1, shuffle=False)
test_datalist = read_from_annfile(img_dir, annfile, y_range=(1, 1000), mode=mode, stack_length=stack_length, ordinal=True)
test_dataset = build_dataset_from_slices(*test_datalist, batch_size=1, parse_func=parse_function, shuffle=False)
evaluation = model.evaluate(test_dataset)

# %% Prediction on each untrimmed videos (test set)
video_list = [vp.stem for vp in Path(img_dir).iterdir()]
predictions = {}
ground_truth = {}
for v in video_list:
    img_list = find_imgs(Path(img_dir, v), stack_length=stack_length)
    ds = build_dataset_from_slices(img_list, batch_size=1, parse_func=parse_function, shuffle=False)
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

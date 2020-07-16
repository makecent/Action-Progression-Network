#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : predict_multi-task.py
# Author: Chongkai LU
# Date  : 16/7/2020

import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt
from load_data import *
from utils import *

# Define names and paths
pretrain = 'scratch'  # or 'imagenet'
model_path = "/mnt/louis-consistent/Saved/DFMAD-70_output/multi/Model/2020-07-14-19-27-08/07-9.48.h5"

img_dir = "/mnt/louis-consistent/Datasets/DFMAD-70/Images/test"
annfile_dir = "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test"

# %% Load trained model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(loss=multi_mse, metrics=[multi_mae])

# %% Evaluation on test trimmed video clips

task1_test_datalist = read_from_anndir(img_dir, annfile_dir, y_range=(0, 100), cumstom_actions=('task1',),
                                       mode='rgb', ordinal=False, weighted=False, stack_length=1)
task2_test_datalist = read_from_anndir(img_dir, annfile_dir, y_range=(0, 100), cumstom_actions=('task2',),
                                       mode='rgb', ordinal=False, weighted=False, stack_length=1)

task1_test_dataset = build_dataset_from_slices(*task1_test_datalist, batch_size=1, shuffle=False)
task2_test_dataset = build_dataset_from_slices(*task2_test_datalist, batch_size=1, shuffle=False)

task1_evaluation = model.evaluate(task1_test_dataset)
task2_evaluation = model.evaluate(task2_test_dataset)

# %% Prediction on test untrimmed videos
video_list = [vp.stem for vp in Path(img_dir).iterdir()]
predictions = {}
ground_truth = {}
for v in video_list:
    img_list = find_imgs(Path(img_dir, v))
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.squeeze(prediction)

# Plot predictions
v = video_list[3]
plt.figure()
plt.plot(predictions[v][:, 0], 'k-', label='task1')
plt.plot(predictions[v][:, 1], 'b-', label='task2')
plt.legend()
plt.show()

# %% Save predictions
with open('saved/task1_{}_multi-task_pre'.format(pretrain), 'w') as f:
    list_pre = {str(k): v[:, 0].tolist() for k, v in predictions.items()}
    json.dump(list_pre, f)

with open('saved/task2_{}_multi-task_pre'.format(pretrain), 'w') as f:
    list_pre = {str(k): v[:, 1].tolist() for k, v in predictions.items()}
    json.dump(list_pre, f)
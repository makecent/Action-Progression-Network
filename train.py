#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 12/7/2020

from load_data import *
from utils import *
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import wandb
from wandb.keras import WandbCallback
import datetime
import tensorflow as tf
import socket

agent = socket.gethostname()
AUTOTUNE = tf.data.experimental.AUTOTUNE
fix_bug()
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# %% wandb Initialization
default_config = dict(
    loss='mse',
    y_s=0,
    y_e=100,
    batch_size=32,
    epochs=10,
    action="task2",
    agent=agent
)
wandb.init(project="dfmad70", config=default_config, name=now, notes='task2 from scratch true')
config = wandb.config
wandbcb = WandbCallback(monitor='val_n_mae', save_model=False)

loss = config.loss
y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0] + 1
batch_size = config.batch_size
epochs = config.epochs
action = config.action

# %% Parameters, Configuration, and Initialization
model_name = now
root = {'train': "/mnt/louis-consistent/Datasets/DFMAD-70/Images/train",
        'test': "/mnt/louis-consistent/Datasets/DFMAD-70/Images/test"}

annfile = {
    'train': "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/train/{}.csv".format(action),
    'test': "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/{}.csv".format(action)}



output_path = '/mnt/louis-consistent/Saved/DFMAD-70_output'  # Directory to save model and history
history_path = Path(output_path, action, 'History', model_name)
models_path = Path(output_path, action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset

datalist = {x: read_from_annfile(root[x], annfile[x], y_range=y_range, mode='rgb', ordinal=False, weighted=False,
                                 stack_length=1) for x in ['train', 'test']}

train_dataset = build_dataset_from_slices(*datalist['train'], batch_size=batch_size, augment=None)
test_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, shuffle=False)

STEP_SIZE_TRAIN = tf.data.experimental.cardinality(train_dataset).numpy()
# %% Build and compile model
n_mae = normalize_mae(y_nums)  # make mae loss normalized into range 0 - 100.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    backbone = ResNet50(weights=None, input_shape=(224, 224, 3), pooling='avg', include_top=False)
    x = backbone(backbone.input)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='relu', kernel_initializer='he_uniform')(x)
    model = Model(backbone.input, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_n_mae:.2f}.h5')), period=1)
    lr_sche = LearningRateScheduler(lr_schedule)
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.0001, decay=1e-3 / STEP_SIZE_TRAIN), metrics=[n_mae])
    his = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs,
                          callbacks=[model_checkpoint, wandbcb, lr_sche], verbose=1)

# %% Save history to csv and images
history = his.history
save_history(history_path, history)
plot_history(history_path, history)

# %% Prediction on untrimmed videos
import pandas as pd
import numpy as np
temporal_annotation = pd.read_csv(annfile['test'], header=None)
video_names = temporal_annotation.iloc[:, 0].unique()
predictions = {}
ground_truth = {}
for v in video_names:
    gt = temporal_annotation.loc[temporal_annotation.iloc[:, 0] == v].iloc[:, 1:].values
    ground_truth[v] = gt

    video_path = Path(root['test'], v)
    img_list = find_imgs(video_path)
    ds = build_dataset_from_slices(img_list, batch_size=1, shuffle=False)
    prediction = model.predict(ds, verbose=1)
    predictions[v] = np.squeeze(prediction)

# %% Detect actions
import numpy as np

action_detected = {}
tps = {}
for v, prediction in predictions.items():
    ads = action_search(prediction, min_T=80, max_T=10, min_L=500)
    action_detected[v] = ads
    tps[v] = calc_truepositive(ads, ground_truth[v], 0.5)

num_gt = sum([len(gt) for gt in ground_truth.values()])
loss = np.vstack(list(action_detected.values()))[:, 2]
tp_values = np.hstack(list(tps.values()))
ap = average_precision(tp_values, num_gt, loss)

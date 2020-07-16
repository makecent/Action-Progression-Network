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
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# %% wandb Initialization
#######
# if you don't want to use wandb for recording, just remove the sub-section and set parameters manually.
#######
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
root = "/mnt/louis-consistent/Datasets/DFMAD-70/Images/train"
annfile = "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/train/{}.csv".format(action)

output_path = '/mnt/louis-consistent/Saved/DFMAD-70_output'  # Directory to save model and history
history_path = Path(output_path, action, 'History', model_name)
models_path = Path(output_path, action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset
datalist = read_from_annfile(root, annfile, y_range=y_range, stack_length=1)
dataset = build_dataset_from_slices(*datalist, batch_size=batch_size, shuffle=True)

data_size = tf.data.experimental.cardinality(dataset).numpy()
val_dataset = dataset.take(int(0.3 * data_size))
train_dataset = dataset.skip(int(0.3 * data_size))
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
    his = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,
                    callbacks=[model_checkpoint, wandbcb, lr_sche], verbose=1)

# %% Save history to csv and images
history = his.history
save_history(history_path, history)
plot_history(history_path, history)

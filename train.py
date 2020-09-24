#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : train.py
# Author: Chongkai LU
# Date  : 12/7/2020

from load_data import *
from utils import *
from pathlib import Path
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from Flated_Inception import Inception_Inflated3d
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
    y_s=1,
    y_e=100,
    batch_size=32,
    stack_length=10,
    epochs=50,
    action="task1",
    mode='rgb',
    pretrain=True,
    agent=agent
)
wandb.init(project="dfmad70", config=default_config, name=now, notes='task1 rgb i3d')
config = wandb.config
wandbcb = WandbCallback(monitor='val_mae_od', save_model=False)
# Configurations. If you don't use wandb, manually set below variables.
y_range = (config.y_s, config.y_e)
y_nums = y_range[1] - y_range[0] + 1
stack_length = config.stack_length
batch_size = config.batch_size
epochs = config.epochs
action = config.action
mode = config.mode
pretrain = config.pretrain

ordinal = True
weighted = False
# Configurations. If you don't use wandb, manually set above variables.
tags = [action, mode, 'i3d']
if ordinal:
    tags.append("od")
if weighted:
    tags.append("weighted")
if stack_length > 1:
    tags.append("stack{}".format(stack_length))

wandb.run.tags = tags
wandb.run.notes = 'i3d_{}_{}'.format(action, mode)
wandb.run.save()
wandbcb = WandbCallback(monitor='val_mae_od', save_model=False)
# %% Parameters, Configuration, and Initialization
model_name = now
root = {'train': "/mnt/louis-consistent/Datasets/DFMAD-70/Images/train",
        'test': "/mnt/louis-consistent/Datasets/DFMAD-70/Images/test"}
annfile = {'train': "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/train/{}.csv".format(action),
           'test': "/mnt/louis-consistent/Datasets/DFMAD-70/Annotations/test/{}.csv".format(action)}

output_path = '/mnt/louis-consistent/Saved/DFMAD-70_output'  # Directory to save model and history
history_path = Path(output_path, action, 'History', model_name)
models_path = Path(output_path, action, 'Model', model_name)
history_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

# %% Build dataset
parse_function = parse_builder(i3d=True, mode=mode)
datalist = {x: read_from_annfile(root[x], annfile[x], y_range=y_range, stack_length=stack_length, ordinal=True) for x in ('train', 'test')}

train_dataset = build_dataset_from_slices(*datalist['train'], batch_size=batch_size, parse_func=parse_function, shuffle=True)
val_dataset = build_dataset_from_slices(*datalist['test'], batch_size=batch_size, parse_func=parse_function, shuffle=False)

STEP_SIZE_TRAIN = tf.data.experimental.cardinality(train_dataset).numpy()
# %% Build and compile model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    backbone = Inception_Inflated3d(
        include_top=False,
        weights='{}_imagenet_and_kinetics'.format(mode if mode == 'rgb' else 'flow') if pretrain is not None else None,
        input_shape=(stack_length, 224, 224, 3 if mode == 'rgb' else 2))
    x = tf.keras.layers.Reshape((1024,))(backbone.output)
    x = Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, kernel_initializer='he_uniform', use_bias=False)(x)
    x = BiasLayer(y_nums)(x)
    output = Activation('sigmoid')(x)
    model = Model(backbone.input, output)
    model_checkpoint = ModelCheckpoint(str(models_path.joinpath('{epoch:02d}-{val_mae_od:.2f}.h5')), period=1)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001, decay=1e-3 / STEP_SIZE_TRAIN), metrics=[mae_od])
    his = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[model_checkpoint, wandbcb], verbose=1)

# %% Save history to csv and images
history = his.history
save_history(history_path, history)
plot_history(history_path, history)

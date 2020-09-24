#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : utils.py
# Author: Chongkai LU
# Date  : 12/7/2020
import tensorflow as tf


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, units, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=(input_shape[-1], self.units),
                                    initializer='zeros',
                                    trainable=True)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'units': self.units})
        return config

    def call(self, inputs):
        return tf.repeat(tf.expand_dims(inputs, -1), self.units, axis=-1) + self.bias


def normalize_mae(y_nums):
    """
    Calculate MAE loss and normalize it to range 0 to 100.
    :param y_nums:
        Float. The original MAE length.
    :return:
        Function. Used as a loss function for keras. While it returns normalized mae loss.
    """
    from tensorflow.keras.losses import mean_absolute_error

    def n_mae(*args, **kwargs):
        mae = mean_absolute_error(*args, **kwargs)
        return mae / (y_nums - 1) * 100.0

    return n_mae


def plot_history(path, history, keys=None):
    from matplotlib import pyplot as plt
    import numpy as np
    from pathlib import Path

    if keys is None:
        keys = list(history.keys())
    if isinstance(path, str):
        path = Path(path)

    for i, key in enumerate(keys):
        plt.figure(figsize=(15, 5))
        if 'val' in key:
            continue
        train = plt.plot(history[key], label='train ' + key.title())
        if 'val_{}'.format(key) in history:
            plt.plot(history['val_{}'.format(key)], '--', color=train[0].get_color(), label='Val ' + key.title())
        plt.legend()
        plt.xticks(np.arange(0, len(history[key]) + 1, 5.0))
        # plt.xlim([0, max(history.epoch)])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(path.joinpath('{}.png'.format(key)))


def save_history(path, history):
    import pandas as pd
    history_pd = pd.DataFrame(data=history)
    history_pd.to_csv(path.joinpath('history.csv'), index=False)


def lr_schedule(epoch, lr):
    # if epoch == 20:
    #     return lr * 0.2
    # if epoch == 40:
    #     return lr * 0.2
    # else:
    return lr


def plot_detection(video_prediction, gt, ads):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    plt.plot(video_prediction, '-')
    plt.vlines(gt[:, 0], 0, 100, colors='r', linestyles='solid', label='ground truth')
    plt.vlines(gt[:, 1], 0, 100, colors='r', linestyles='solid', label='ground truth')
    plt.vlines(ads[:, 0], 0, 100, colors='k', linestyles='dashed', label='ground truth')
    plt.vlines(ads[:, 1], 0, 100, colors='k', linestyles='dashed', label='ground truth')
    plt.yticks(np.arange(0, 100, 20.0))
    plt.xlabel('Frame Index')
    plt.ylabel('Completeness')
    plt.grid()
    plt.show()


def plot_prediction(video_prediction):
    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure(figsize=(15, 5))
    plt.plot(video_prediction, '-')
    plt.yticks(np.arange(0, 100, 20.0))
    plt.xlabel('Frame Index')
    plt.ylabel('Completeness')
    plt.grid()
    plt.show()


def iou(a, b):
    ov = 0
    union = max(a[1], b[1]) - min(a[0], b[0])
    intersection = min(a[1], b[1]) - max(a[0], b[0])
    if intersection > 0:
        ov = intersection / union
    return ov


def matrix_iou(gt, ads):
    import numpy as np
    ov_m = np.zeros([gt.shape[0], ads.shape[0]])
    for i in range(gt.shape[0]):
        for j in range(ads.shape[0]):
            ov_m[i, j] = iou(gt[i, :], ads[j, :])
    return ov_m


def calc_truepositive(action_detected, temporal_annotations, iou_T):
    """
    Give the predicted action intervals and ground truth intervals, using IoU threshold to get true positive proposals.
    :param action_detected: Array. Shape (N, 4). Float. 1st and 2st columns contain start and ending frame indexes of detected actions.
            3st column for confidence/loss. Here is for loss, which means tp will be sort with ascend order. 4st for action index.
    :param temporal_annotations: Array. Shape (M, 2). Float. 1st and 2st columns contain start and ending frame indexes of ground truthes.
    :param iou_T: Float. IoU thredhold. A detected action can be true positive only if it has IoU larger than threhold with a ground truth.
    :return: Array. Shape (N,). Composing 0 and 1 corresponding to each detected action, 1 means true positive.
    """
    import numpy as np
    num_detection = action_detected.shape[0]
    if num_detection == 0:
        return np.array([], dtype=np.int)
    iou_matrix = matrix_iou(temporal_annotations, action_detected[:, :2])
    tp = np.zeros(num_detection, dtype=np.int)
    while iou_matrix.size > 0:
        max_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        if iou_matrix[max_idx] > iou_T:
            iou_matrix[:, max_idx[1]] = 0
            iou_matrix[max_idx[0], :] = 0
            tp[max_idx[1]] = 1
        else:
            break
    return tp


def average_precision(tp, num_gt, loss):
    """
    Compute average precision with given true positive indicator and number of ground truth.
    :param tp: Array. Shape (N,). Comprising 0 and 1. Represents the T or F of proposed predictions.
    :param num_gt: Int. Number of ground truth samples.
    :param loss: Array. Shape (N,). loss of each predictions used for sort the tp. For using confidence, code need to be revised.
    :return: Array. Shape (1,). Average Precision.
    """
    import numpy as np
    tp = tp[loss.argsort()]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    precisions = cum_tp / (cum_tp + cum_fp)
    AP = np.sum(precisions * tp) / num_gt
    return AP


def multi_mse(y_true, y_pred):
    import tensorflow as tf
    # indexing row of y_pred by action index stored in y_true. [batch_size, 2] --> [batch_size]
    y_pred = tf.gather_nd(y_pred, tf.expand_dims(tf.cast(y_true[:, 0], tf.int32), axis=-1), batch_dims=1)
    # convert int completeness to ordinal vector. [batch_size, 2] --> [batch_size]
    y_true = tf.squeeze(y_true[:, 1])
    y_true = tf.cast(y_true, tf.float32)

    multi_mse_loss = tf.math.square(tf.math.abs(y_true - y_pred))
    return multi_mse_loss


def multi_mae(y_true, y_pred):
    # if not use y_range 0-100, please normalize this metric manually for better visualization and comparison.
    import tensorflow as tf
    # indexing row of y_pred by action index stored in y_true. [batch_size, 2] --> [batch_size]
    y_pred = tf.gather_nd(y_pred, tf.expand_dims(tf.cast(y_true[:, 0], tf.int32), axis=-1), batch_dims=1)
    # convert int completeness to ordinal vector. [batch_size, 2] --> [batch_size]
    y_true = tf.squeeze(y_true[:, 1])
    y_true = tf.cast(y_true, tf.float32)

    multi_mae_loss = tf.math.abs(y_true - y_pred)
    return multi_mae_loss


def mae_od(y_true, y_pred):
    import tensorflow as tf
    predict_completeness = tf.math.count_nonzero(y_pred > 0.5, axis=-1)
    true_completeness = tf.math.count_nonzero(y_true > 0.5, axis=-1)
    mean_absolute_error = tf.math.abs(predict_completeness - true_completeness)
    return mean_absolute_error


def action_search(completeness_array, min_T, max_T, min_L):
    import numpy as np
    """
    Detect (temporal localization) complete action on completeness list.
    :param completeness_array: Numpy Array. List of float numbers, completeness of frames
    :param min_T: Int. Minimum completeness value threshold used to find end frame candidates.
    :param max_T: Int. Maximum completeness value threshold used to find start frame candidates.
    :param min_L: Int. Minimum complete action length used
    :return: List. List of list. each list represent a detected action illustrated as [start_inx(int) end_inx(int) loss(float)]
    Examples:
    min_T, max_T, min_L = 75, 20, 35
    """

    def is_intersect(a, b):
        if a[0] > b[1] or a[1] < b[0]:
            return False
        else:
            return True

    P = completeness_array.squeeze()
    C_startframe = np.where(P < max_T)[0]  # "C_" represent variable for candidates.
    C_endframe = np.where(P > min_T)[0]
    action_detected = []
    for s_i in C_startframe:
        for e_i in C_endframe:
            C_action_length = e_i - s_i + 1
            if C_action_length > min_L:
                action_template = np.linspace(0, 100, C_action_length)
                predicted_sequence = P[s_i:e_i + 1]
                mse = ((action_template - predicted_sequence) ** 2).mean()
                action_candidate = [s_i, e_i, mse]
                any_intersection = False
                beat_any_one = False
                for i, action in enumerate(action_detected):
                    if is_intersect(action, action_candidate):
                        any_intersection = True
                        if action_candidate[2] < action[2]:
                            beat_any_one = True
                            action_detected.pop(i)
                if beat_any_one or not any_intersection:
                    action_detected.append(action_candidate)
    action_detected.sort(key=lambda x: x[2])
    return np.array(action_detected).reshape(-1, 3)

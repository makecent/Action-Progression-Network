#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : load_data.py
# Author: Chongkai LU
# Date  : 12/7/2020

# %% Special Function: Can only be used in this program. Most of them aim to make main file more concise.
def format_img(image, label=None, weight=None):
    import tensorflow as tf
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (224, 224))
    if label is None:
        return image
    elif weight is None:
        return image, label
    else:
        return image, label, weight


def generate_labels(length, y_range, multi_action=False, action_index=None):
    """
    Given number of frames of a trimmed action video. Return a label array monotonically increasing in y_range.
    Mainly based on numpy.linspace function
    :param length:
        Int. Number of completeness labels need to be generate.
    :param y_range:
        Tuple. Range of the completeness label values vary in.
    :param multi_action:
        Boolean. Since in multi_task case ordinal label vector will be too large (20 X 100), hence take too much space
        to store. Therefore in this case each label will be set as single value even ordinal is True. The vector
        transfer will be conducted on loss and metric function instead. To identify the action class, each label will be
        along with a action index.
    :param action_index:
        Int. Identify the action class of correspond label
    :return:
        Array.
    """
    import numpy as np
    completeness = np.linspace(*y_range, num=length, dtype=np.float32)
    if multi_action:
        return np.insert(completeness[..., np.newaxis], 0, action_index, axis=1)
    else:
        return completeness


def read_from_annfile(root, annfile, y_range, stack_length=1, multi_action=False, action_index=None):
    """
    According to the temporal annotation file of a action. Create list of image paths and list of labels.
    :param root: String. Directory where all images are stored.
    :param annfile: String. Path where temporal annotation locates.
    :param y_range: Tuple. Label range.
    :param stack_length: Int. Number of input images stacked for training unit. Default 1 for RGB,
    :param multi_action: Boolean. If true, label will be assign a action_index along aside for indicating its action class
    :param action_index: Int. Only used when multi_action is True, representing the action_index for this action.
    :return: List, List. List of image paths and list of labels and list of weights (if weighted=True).
    """
    import pandas as pd
    import numpy as np
    temporal_annotations = pd.read_csv(annfile, header=None)

    stacked_list, labels, weights = [], [], []
    for video, start, end in temporal_annotations.itertuples(index=False):
        v_paths = ["{}/{}/{}.jpg".format(root, video, str(num).zfill(5)) for num in np.arange(start, end + 1)]
        v_stacked_paths = [v_paths[i:i + stack_length] for i in range(0, len(v_paths) - stack_length + 1)]
        stacked_list.extend(v_stacked_paths)

        v_stacked_length = len(v_stacked_paths)
        labels.extend(generate_labels(v_stacked_length, y_range, multi_action, action_index))

    return stacked_list, labels


def read_from_anndir(root, anndir, cumstom_actions=None, **kwargs):
    """
    Warped function for "read_from_annfile" to read multiple annfiles in an directory for multiple actions.
    According to the temporal annotation files. Create list of image paths and list of labels.
    :param root: String. Directory where all images are stored.
    :param anndir: String. Path where temporal annotation locates.
    :param cumstom_actions: List of String. Contain name of actions that need to read. If None, then all action
    annotation files in this dir will be used.
    :return: List, List. List of image paths and list of labels.
    """
    from pathlib import Path

    action_idx = {'task1': 0, 'task2': 1}

    datalist, ylist = [], []
    for annfile in sorted(Path(anndir).iterdir()):
        action_name = str(annfile.stem).split('.')[0]
        if cumstom_actions:
            if action_name not in cumstom_actions:
                continue
        action_list, label_list = read_from_annfile(root, str(annfile), multi_action=True, action_index=action_idx[action_name], **kwargs)
        datalist.extend(action_list)
        ylist.extend(label_list)
    return datalist, ylist


# %% Basic function: Can be easily used in other programs.
def decode_img(file_path, label=None, weight=None):
    """
    Read image from path
    :param label: Unknown.
    :param file_path: String.
    :return: Image Tensor.
    """
    import tensorflow as tf
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    if label is None:
        return img
    elif weight is None:
        return img, label
    else:
        return img, label, weight


def build_dataset_from_slices(data_list, labels_list=None, batch_size=32, augment=None, shuffle=True,
                              prefetch=True):
    """
    Given image paths and labels, create tf.data.Dataset instance.
    :param data_list: List. Consists of strings. Each string is a path of one image.
    :param labels_list: List. Consists of labels. None means for only prediction.
    :param batch_size: Int.
    :param augment: Func. Data augment function. input: (stacked_imgs, **kwargs), output (augmented_imgs, **kwargs)
    :param shuffle: Boolean. True for train, False for prediction and evaluation
    :param prefetch: Boolean. Refer to prefetch technology in tensorflow.data.Dataset
    :return: tf.data.Dataset.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if labels_list is None:
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_list))
    if shuffle:
        dataset = dataset.shuffle(len(data_list))

    def stack_decode_format(filepath_list, labels=None):
        """Decode stacked image paths to stacked image tensors and format to desired format"""
        filepath_list = tf.unstack(filepath_list, axis=-1)
        flow_snip = []
        for flow_path in filepath_list:
            decoded = decode_img(flow_path)
            flow_snip.append(format_img(decoded))
        parsed = tf.concat(flow_snip, axis=-1)
        if labels is None:
            return parsed
        else:
            return parsed, labels

    dataset = dataset.map(stack_decode_format, num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    if batch_size > 0:
        dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def find_imgs(video_path, suffix='jpg', stack_length=1):
    """
    Find all images in a given path.
    :param video_path: String. Target video folder
    :return: List. Consists of strings. Each string is a image path; Sorted.
    """
    from pathlib import Path
    if isinstance(video_path, str):
        video_path = Path(video_path)
    imgs_list = [str(jp) for jp in sorted(video_path.glob('*.{}'.format(suffix)))]
    stacked_imgs_list = [imgs_list[i:i + stack_length] for i in range(0, len(imgs_list) - stack_length + 1)]
    return stacked_imgs_list

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : load_data.py
# Author: LU Chongkai
# Date  : 23/5/2019

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


def generate_labels(length, y_range, ordinal=False, multi_action=False, action_index=0):
    """
    Given number of frames of a trimmed action video. Return a label array monotonically increasing in y_range.
    Mainly based on numpy.linspace function
    :param length:
        Int. Number of completeness labels need to be generate.
    :param y_range:
        Tuple. Range of the completeness label values vary in.
    :param ordinal:
        Boolean. If True, then each completeness label will be convert to a ordinal vector. e.g. 3 -> [1,1,1,0,0,...]
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
    y_nums = y_range[1] - y_range[0] + 1
    completeness = np.linspace(*y_range, num=length, dtype=np.float32)
    if ordinal:
        rounded_completeness = np.round(completeness).astype(np.int)
        ordinal_completeness = np.array([[1] * int(c) + [0] * int(y_nums - c) for c in rounded_completeness],
                                        dtype=np.float32)
        ordinal_completeness = np.expand_dims(ordinal_completeness, axis=-2)
        if multi_action:
            return np.expand_dims(np.insert(rounded_completeness[..., np.newaxis], 0, action_index, axis=1), axis=-1)
        else:
            return ordinal_completeness
    else:
        if multi_action:
            return np.insert(completeness[..., np.newaxis], 1, action_index, axis=1)
        else:
            return completeness


def path_rgb2flow(rgb_path):
    import tensorflow as tf
    t = tf.strings.regex_replace(rgb_path, 'Images', 'OpticalFlows', replace_global=True, name=None)
    f_x = tf.strings.regex_replace(t, '[0-9]+.jpg', 'flow_x/flow_x_', replace_global=True, name=None)
    f_y = tf.strings.regex_replace(t, '[0-9]+.jpg', 'flow_y/flow_y_', replace_global=True, name=None)
    a = tf.strings.substr(t, -9, 5)
    a = tf.strings.to_number(a, out_type=tf.dtypes.int64) + 1
    a = tf.strings.as_string(a, width=5, fill='0')
    b = tf.constant('.jpg')
    flow_x_path = f_x+a+b
    flow_y_path = f_y+a+b
    return tf.stack([flow_x_path, flow_y_path])


def read_from_annfile(root, annfile, mode='rgb', stack_length=1, weighted=False, **kwargs):
    """
    According to the temporal annotation file of a action. Create list of image paths and list of labels.
    :param root. String. Path where all images locate.
    :param annfile: String. Path where temporal annotation locates.
    :param mode: String. Optical flow or RGB mode.
    :param stack_length: Int. Number of input images stacked for training unit. Default 1 for RGB,
    For optical flow, default 10 (finally will get 20 since both u and v).
    :param weighted: Boolean. If true, the samples with completeness in two side will be put more weights and this function
    will also return a sample-wise weights list.
    :param kwargs: Dict. Arguments passed to function "generate_labels"
    :return: List, List. List of image paths and list of labels and list of weights (if weighted=True).
    """
    import pandas as pd
    import numpy as np
    temporal_annotations = pd.read_csv(annfile, header=None)

    stacked_list, labels, weights = [], [], []
    for video, start, end in temporal_annotations.itertuples(index=False):
        if end-start < stack_length:
            continue
        if mode == 'rgb':
            v_paths = ["{}/{}/{}.jpg".format(root, video, str(num).zfill(5)) for num in np.arange(start, end)]
            v_stacked_paths = [v_paths[i:i + stack_length] for i in range(0, len(v_paths) - stack_length + 1)]
        else:
            v_paths = ["{}/{}/{}/{}_{}.jpg".format(root, video, d, d, str(num + 1).zfill(5)) for num in
                       np.arange(start, end) for d in ['flow_x', 'flow_y']]
            v_stacked_paths = [v_paths[2 * i:2 * i + stack_length * 2] for i in
                               range(0, len(v_paths) // 2 - stack_length + 1)]
        stacked_list.extend(v_stacked_paths)
        v_stacked_length = len(v_stacked_paths)
        labels.extend(generate_labels(v_stacked_length, **kwargs))

        if weighted:
            w_10 = [3, 2, 1, 1, 1, 1, 1, 1, 2, 3]
            weights.extend(np.hstack([w * p for w, p in zip(w_10, np.array_split(np.ones(v_stacked_length), 10))]))

    return (stacked_list, labels, weights) if weighted else (stacked_list, labels)


def read_from_anndir(root, anndir, **kwargs):
    """
    Warped function for "read_from_annfile" to read multiple annfiles in an directory for multiple actions.
    According to the temporal annotation files. Create list of image paths and list of labels.
    :param root. String. Path where all images locate.
    :param anndir: String. Path where temporal annotation locates.
    :param kwargs: Dict. Arguments passed to function "generate_labels"
    :return: List, List. List of image paths and list of labels.
    """
    from pathlib import Path

    action_idx = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 'CleanAndJerk': 3, 'CliffDiving': 4,
                  'CricketBowling': 5, 'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 'GolfSwing': 9,
                  'HammerThrow': 10, 'HighJump': 11, 'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 'Shotput': 15,
                  'SoccerPenalty': 16, 'TennisSwing': 17, 'ThrowDiscus': 18, 'VolleyballSpiking': 19}

    datalist, ylist = [], []
    for annfile in sorted(Path(anndir).iterdir()):
        action_name = str(annfile.stem).split('_')[0]
        action_list, label_list = read_from_annfile(root=root, annfile=str(annfile), multi_action=True,
                                                    action_index=action_idx[action_name], **kwargs)
        datalist.extend(action_list)
        ylist.extend(label_list)
    return datalist, ylist


def parse_builder(mode='rgb', i3d=False):
    import tensorflow as tf

    def i3d_stack_decode_format(filepath_list, labels=None, weights=None):
        """Decode stacked image paths to stacked image tensors and format to desired format"""
        filepath_list = tf.unstack(filepath_list, axis=-1)
        flow_snip = []
        for flow_path in filepath_list:
            decoded = decode_img(flow_path)
            flow_snip.append(format_img(decoded))
        parsed = tf.stack(flow_snip, axis=0)
        if labels is None:
            return parsed
        elif weights is None:
            return parsed, labels
        else:
            return parsed, labels, weights

    def i3d_stack_flow_decode_format(filepath_list, labels=None, weights=None):
        """Decode stacked image paths to stacked image tensors and format to desired format"""
        filepath_list = tf.unstack(filepath_list, axis=-1)
        flow_snip = []
        for flow_x_path, flow_y_path in zip(filepath_list[::2], filepath_list[1::2]):
            decoded_x = decode_img(flow_x_path)
            decoded_y = decode_img(flow_y_path)
            decoded_flow = tf.concat([decoded_x, decoded_y], axis=-1)
            flow_snip.append(format_img(decoded_flow))
        parsed = tf.stack(flow_snip, axis=0)
        if labels is None:
            return parsed
        elif weights is None:
            return parsed, labels
        else:
            return parsed, labels, weights

    def i3d_two_stream_decode_format(filepath_list, labels=None, weights=None):
        """Decode stacked image paths to stacked image tensors and format to desired format"""
        rgb = i3d_stack_decode_format(filepath_list)

        flow_list = tf.map_fn(path_rgb2flow, filepath_list)
        flow_list = tf.reshape(flow_list, [-1])
        flow = i3d_stack_flow_decode_format(flow_list)
        if labels is None:
            return {'rgb_input': rgb, 'flow_input': flow}
        elif weights is None:
            return {'rgb_input': rgb, 'flow_input': flow}, labels
        else:
            return {'rgb_input': rgb, 'flow_input': flow}, labels, weights

    def stack_decode_format(filepath_list, labels=None, weights=None):
        """Decode stacked image paths to stacked image tensors and format to desired format"""
        filepath_list = tf.unstack(filepath_list, axis=-1)
        flow_snip = []
        for flow_path in filepath_list:
            decoded = decode_img(flow_path)
            flow_snip.append(format_img(decoded))
        parsed = tf.concat(flow_snip, axis=-1)
        if labels is None:
            return parsed
        elif weights is None:
            return parsed, labels
        else:
            return parsed, labels, weights

    if i3d:
        if mode == 'rgb':
            parse_function = i3d_stack_decode_format
        elif mode == 'flow' or mode == 'w_flow':
            parse_function = i3d_stack_flow_decode_format
        elif mode == 'two_stream':
            parse_function = i3d_two_stream_decode_format
        else:
            raise TypeError(
                "Please input 'mode' type in (rgb, flow, w_flow, two_stream) for function 'build_dataset_from_slices'")
    else:
        parse_function = stack_decode_format

    return parse_function


# %% Basic function: Can be used in other programs.


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


def decode_img_relative(root):
    import os
    def decode_img_with_root(file_path, label=None, weight=None):
        absolute_paths = [os.path.join(root, p) for p in file_path]
        return decode_img(absolute_paths, label, weight)

    return decode_img_with_root


def build_dataset_from_slices(data_list, labels_list=None, weighs=None, parse_func=None, batch_size=32, augment=None,
                              shuffle=True, prefetch=True):
    """
    Given image paths and labels, create tf.data.Dataset instance.
    :param data_list: List. Consists of strings. Each string is a path of one image.
    :param labels_list: List. Consists of labels. None means for only prediction.
    :param weighs: List. Consists of sample-wise weigths. None for prediction.
    :param batch_size: Int.
    :param augment: Func. Data augment function. input: (stacked_imgs, labels, weights), output (augmented_imgs, labels, weights)
    :param shuffle: Boolean. True for train, False for prediction and evaluation
    :param prefetch: Boolean. Refer to prefetch technology in tensorflow.data.Dataset
    :return: tf.data.Dataset.
    """
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if labels_list is None:
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
    elif weighs is None:
        dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_list))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((data_list, labels_list, weighs))
    if shuffle:
        dataset = dataset.shuffle(len(data_list))

    dataset = dataset.map(parse_func)

    if augment:
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    if batch_size > 0:
        dataset = dataset.batch(batch_size, drop_remainder=True)
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
    del imgs_list[-1]  # for alignment with optical flow images.
    stacked_imgs_list = [imgs_list[i:i + stack_length] for i in range(0, len(imgs_list) - stack_length + 1)]
    return stacked_imgs_list


def find_flows(video_path, suffix='jpg', stack_length=10):
    """
    Find all images in a given path.
    :param video_path: String. Target video folder
    :return: List. Consists of strings. Each string is a image path; Sorted.
    """
    from pathlib import Path
    if isinstance(video_path, str):
        video_path = Path(video_path)
    flow_x = sorted(video_path.glob('flow_x/*.{}'.format(suffix)))
    flow_y = sorted(video_path.glob('flow_y/*.{}'.format(suffix)))
    v_fl = [str(e) for xy in zip(flow_x, flow_y) for e in xy]
    v_stacked_flow = [v_fl[2 * i:2 * i + stack_length * 2] for i in range(0, len(v_fl) // 2 - stack_length + 1)]
    return v_stacked_flow

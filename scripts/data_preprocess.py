import os

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

from scripts.model import data_file_name

len_window = 100
len_train_set = 0.85


def distributor(x, y, len_train_set):
    train_x, train_y = [], []
    test_x, test_y = [], []

    quantity = label_check(y)

    min_val = min(quantity.values())

    len_train_set = int(min_val * len_train_set)

    pos_label = 0
    neg_label = 0

    for idx, sample in enumerate(x):

        label = y[idx]

        if label == 0 and neg_label < len_train_set:

            neg_label += 1
            train_x.append(sample)
            train_y.append(label)

        elif label == 1 and pos_label < len_train_set:

            pos_label += 1
            train_x.append(sample)
            train_y.append(label)

        else:

            test_x.append(sample)
            test_y.append(label)

    return (np.array(i) for i in [train_x, train_y, test_x, test_y])


def label_check(labels):
    unique, counts = np.unique(labels, return_counts=True)
    quantity = dict(zip(unique, counts))

    return quantity


def sliding_window(tensor, labels, len_window):
    out_tensor = []
    out_labels = []

    for idx, matrix in enumerate(tensor):

        timesteps = matrix.shape[0]

        if timesteps >= len_window:
            for j in range(matrix.shape[0] - len_window):
                out_tensor.append(matrix[j:j + len_window])
                out_labels.append(labels[idx])

    out_tensor = np.array(out_tensor)
    out_labels = np.array(out_labels)

    return out_tensor, out_labels


def data_collect(x, y, data_folder, label):
    file_names = os.listdir(data_folder)

    for i, name in enumerate(file_names):
        temp_path = os.path.join(data_folder, name)
        file = pd.read_csv(temp_path, ';', names=['times', 'parts', 'x', 'y', 'z'])

        temp_arr = file[['x', 'y', 'z']].to_numpy()
        # temp_arr = np.linalg.norm(array, axis=-1, keepdims=True)
        temp_arr = temp_arr.reshape(-1, 20, 3)

        x.append(temp_arr)
        y.append(label)

    return x, y


if __name__ == "__main__":
    ctrl = '../data/processed_ctrl'
    prksn = '../data/processed_prksn'

    x = []
    y = []

    x, y = data_collect(x, y, ctrl, 0)
    x, y = data_collect(x, y, prksn, 1)

    x, y = np.array(x), np.array(y)

    x, y = shuffle(x, y)

    x, y = sliding_window(x, y, len_window=len_window)

    quantity = label_check(y)

    train_x, train_y, test_x, test_y = distributor(x, y, len_train_set=len_train_set)

    print(f"Длина тренировочного сета = {len(train_x)}")
    print(f"Длина окна = {len_window}")
    print()

    print(f"Здоровых = {quantity[0]}")
    print(f"Больных = {quantity[1]}")
    print()

    print('train_x', train_x.shape)
    print('train_y', label_check(train_y))
    print()

    print('test_x', test_x.shape)
    print('test_y', label_check(test_y))

    np.save(data_file_name, np.array((train_x, train_y, test_x, test_y)))

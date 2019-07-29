import os

import numpy as np 
import pandas as pd 

from keras.preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle

len_window = 100

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

				out_tensor.append(matrix[j:j+len_window])
				out_labels.append(labels[idx])		

	out_tensor = np.array(out_tensor)
	out_labels = np.array(out_labels)

	return out_tensor, out_labels


def data_collect(x, y, data_folder, label):

	file_names = os.listdir(data_folder)

	for i, name in enumerate(file_names):
		temp_path = os.path.join(data_folder, name)
		file = pd.read_csv(temp_path, ';', names = ['times', 'parts', 'x','y','z'])

		array = file[['x','y','z']].to_numpy()
		temp_arr = np.linalg.norm(array, axis=-1, keepdims=True)
		temp_arr = temp_arr.reshape(-1, 20)
		
		x.append(temp_arr)
		y.append(label)

	return x, y



ctrl = '../data/processed_ctrl'
prksn = '../data/processed_prksn'
destination = '../data/tensors'

x = []
y = []

x, y = data_collect(x, y, ctrl, 0)
x, y = data_collect(x, y, prksn, 1)

x, y = np.array(x), np.array(y)

x, y = shuffle(x, y)

x, y = sliding_window(x, y, len_window=len_window)

quantity = label_check(y)

len_train_set = int(len(x) * 0.9)

train_x, train_y = x[:len_train_set], y[:len_train_set]
test_x, test_y = x[len_train_set:], y[len_train_set:]
# train_x, train_y = pad_sequences(train_x, maxlen=300), np.array(train_y)
# test_x, test_y = pad_sequences(test_x, maxlen=300), np.array(test_y)

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

np.save(f'{destination}/tensor_window={len_window}', np.array((train_x, train_y, test_x, test_y)))

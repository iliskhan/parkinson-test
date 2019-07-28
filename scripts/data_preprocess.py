import os

import numpy as np 
import pandas as pd 

from keras.preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle

len_train_set = 16


def sliding_window(tensor, labels, len_window):
	out_tensor = []
	out_labels = []

	for idx, matrix in enumerate(tensor):

		for j in range(matrix.shape[0] - len_window):

			out_tensor.append(matrix[j:j+len_window])
			out_labels.append(labels[idx])
			

	out_tensor = np.array(out_tensor)
	out_labels = np.array(out_labels)

	return out_tensor, out_labels


def data_collect(data_folder, label):

	file_names = os.listdir(data_folder)

	for i, name in enumerate(file_names):
		temp_path = os.path.join(data_folder, name)
		file = pd.read_csv(temp_path, ';', names = ['times', 'parts', 'x','y','z'])

		array = file[['x','y','z']].to_numpy()
		temp_arr = np.linalg.norm(array, axis=-1, keepdims=True)
		temp_arr = temp_arr.reshape(-1, 20)
		
		if i < len_train_set:
			train_x.append(temp_arr)
			train_y.append(label)

		else:
			test_x.append(temp_arr)
			test_y.append(label)



ctrl = '../data/processed_ctrl'
prksn = '../data/processed_prksn'
destination = '../data/tensors'

train_x = []
train_y = []

test_x = []
test_y = []

data_collect(ctrl,0)
data_collect(prksn,1)

train_x, train_y = np.array(train_x), np.array(train_y)
test_x, test_y = np.array(test_x), np.array(test_y)

train_x, train_y = sliding_window(train_x, train_y, len_window=200)
test_x, test_y = sliding_window(test_x, test_y, len_window=200)

# train_x, train_y = pad_sequences(train_x, maxlen=300), np.array(train_y)

# test_x, test_y = pad_sequences(test_x, maxlen=300), np.array(test_y)

train_x, train_y = shuffle(train_x, train_y)
test_x, test_y = shuffle(test_x, test_y)

print('train_x', train_x.shape)
print('test_x', test_x.shape)

print('train_y', train_y.shape)
print('test_y', test_y.shape)

np.save(f'{destination}/tensor6', np.array((train_x, train_y, test_x, test_y)))

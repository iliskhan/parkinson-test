import os

import numpy as np 
import pandas as pd 

from keras.preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle

def sliding_window(arr, len_window):
	new_set = []
	arr = arr.T

	
	for i in arr:
		for j in range(len(i)-len_window):
			new_set.append(i[j:j+len_window])
	new_set = np.array(new_set)
	print(new_set.shape)

def fill_arr(arr, filled_arr, slice=0):

	arr_counter = 0

	if slice == 0:
		for i in range(0, filled_arr.shape[0], 5):
			for j in range(5):
				filled_arr[i+j] = arr[arr_counter]
			arr_counter+=1

	else:
		for i in range(0, filled_arr.shape[0], 5):
			for j in range(5):
				filled_arr[i+j] = arr[arr_counter, j*slice:(1+j)*slice]
			arr_counter+=1

	return filled_arr



def data_collect(data_folder, label):

	file_names = os.listdir(data_folder)

	for i, name in enumerate(file_names):
		temp_path = os.path.join(data_folder, name)
		file = pd.read_csv(temp_path, ';', names = ['times', 'parts', 'x','y','z'])

		array = file[['x','y','z']].to_numpy()
		temp_arr = np.linalg.norm(array, axis=-1, keepdims=True)
		temp_arr = temp_arr.reshape(-1, 20)

		new_arr = sliding_window(temp_arr, 200)
		
		if i < 16:
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

train_x, train_y = pad_sequences(train_x, maxlen=300), np.array(train_y)

test_x, test_y = pad_sequences(test_x, maxlen=300), np.array(test_y)

train_x, train_y = shuffle(train_x, train_y)
test_x, test_y = shuffle(test_x, test_y)

# augment_train_x = np.zeros((train_x.shape[0]*5, max_val//5, 20))
# augment_test_x = np.zeros((test_x.shape[0]*5, max_val//5, 20))

# augment_train_y = np.zeros((train_y.shape[0]*5))
# augment_test_y = np.zeros((test_y.shape[0]*5))

# augment_train_x = fill_arr(train_x, augment_train_x, max_val//5)
# augment_test_x = fill_arr(test_x, augment_test_x, max_val//5)

# augment_train_y = fill_arr(train_y, augment_train_y)
# augment_test_y = fill_arr(test_y, augment_test_y)

print('train_x', train_x.shape)
print('test_x', test_x.shape)

print('train_y', train_y.shape)
print('test_y', test_y.shape)

#np.save(f'{destination}/tensor5', np.array((train_x, train_y, test_x, test_y)))

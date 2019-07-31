import os

import numpy as np 

import keras

from keras import regularizers

from keras.models import Model
from keras.layers import GRU, LSTM, Input, Dropout, Dense, Activation, TimeDistributed, Flatten

len_window = 100

train_x, train_y, test_x, test_y = np.load(f'../data/tensors/tensor_window={len_window}.npy', allow_pickle=True)

l1 = regularizers.l1(0.003)
l2 = regularizers.l2(0.003)

x_input = Input(shape=(len_window, 20, 3))

x = TimeDistributed(Flatten(), input_shape=(20, 3))(x_input)

x = GRU(100, recurrent_dropout=0.3, 
		bias_regularizer=l2, 
		kernel_regularizer=l2, 
		activity_regularizer=l2,
		recurrent_regularizer=l2)(x) 
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(256, kernel_regularizer=l2, 
		  bias_regularizer=l2, 
		  activity_regularizer=l2)(x)
x = Activation('tanh')(x)
x = Dropout(0.5)(x)

x = Dense(1)(x)
x = Activation('sigmoid')(x)

model = Model(inputs=x_input, outputs=x)

model.compile("nadam", loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, epochs=50)

preds = model.evaluate(x=test_x, y=test_y)

model.save(f'../models/parkinson_window={len_window}.h5')

print('Test loss =', preds[0])
print('Test accuracy =', preds[1])


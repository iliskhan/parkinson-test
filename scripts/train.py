import os

import numpy as np 

import keras

from keras.models import Model
from keras.layers import GRU, LSTM, Input, Dropout, Dense, Activation, TimeDistributed


len_window = 100

train_x, train_y, test_x, test_y = np.load(f'../data/tensors/tensor_window={len_window}.npy', allow_pickle=True)

x_input = Input(shape=(len_window, 20))

x = LSTM(100, recurrent_dropout=0.3)(x_input)
x = Activation('relu')(x)
x = Dropout(0.3)(x)

x = Dense(256)(x)
x = Activation('tanh')(x)
x = Dropout(0.5)(x)

x = Dense(1)(x)
x = Activation('sigmoid')(x)

model = Model(inputs=x_input, outputs=x)

model.compile("adam", loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, epochs=20)

preds = model.evaluate(x=test_x, y=test_y)

model.save(f'../models/parkinson_window={len_window}.h5')

print('Test loss =', preds[0])
print('Test accuracy =', preds[1])


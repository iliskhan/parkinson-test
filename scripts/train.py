import os

import numpy as np 

import keras

from keras.models import Model
from keras.layers import GRU, LSTM, Input, Dropout, Dense, Activation, TimeDistributed


train_x, train_y, test_x, test_y = np.load('../data/tensors/tensor5.npy', allow_pickle=True)

x_input = Input(shape=(None, 20))

x = LSTM(64)(x_input)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(1)(x)
x = Activation('sigmoid')(x)

model = Model(inputs=x_input, outputs=x)

model.compile("adam", loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_x, y=train_y, epochs=100)

preds = model.evaluate(x=test_x, y=test_y)

model.save('../models/parkinson.h5')

print('Test loss =', preds[0])
print('Test accuracy =', preds[1])


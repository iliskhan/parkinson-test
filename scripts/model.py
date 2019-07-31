import numpy as np

from keras import regularizers
from keras.models import load_model as keras_load_model

from keras.models import Model
from keras.layers import GRU, LSTM, Input, Dropout, Dense, Activation, TimeDistributed, Flatten

len_window = 100
data_file_name = f'../data/tensors/tensor_window={len_window}.npy'
model_file_name = f'../models/parkinson_window={len_window}.h5'


def load_data():
    train_x, train_y, test_x, test_y = np.load(data_file_name, allow_pickle=True)
    return train_x, train_y, test_x, test_y


def load_model():
    return keras_load_model(model_file_name)


def build_model():
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

    return Model(inputs=x_input, outputs=x)


def compile_model(model):
    model.compile("nadam", loss='binary_crossentropy', metrics=['accuracy'])
    return model

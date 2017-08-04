"""
2D convolutional neural net model.

The idea is we transform the feature vector to a 2D matrix which we'll then use a convnet model to do (hopefully)
automatic feature selection by abstraction of the convolutional layers.
"""


import numpy as np
import pandas as pd


from preprocessor import load_dataset


from sklearn import metrics, preprocessing, linear_model

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def conv_model():
    model = Sequential()
    model.add(Conv2D(16, (2, 2), padding="same", input_shape=(1, 7, 3), data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["binary_accuracy", "mae"])
    return model


def main(grouped=False):
    # prepare the dataset

    if grouped:
        dataset = load_dataset(as_matrix=(1, 7, 3), group_by_era=True)
        train_set = dataset['training']
    else:
        X, Y, x_val, y_val = load_dataset(as_matrix=(1, 7, 3))

    X, Y = train_set[0]
   
    # train the model
    model = conv_model()
    tb_callback = keras.callbacks.TensorBoard(log_dir='./dump', histogram_freq=0,  
          write_graph=True, write_images=True)
    model.fit(X, Y, epochs=100, batch_size=10,  callbacks=[tb_callback])


main(True)

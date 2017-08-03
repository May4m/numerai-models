
import numpy as np
import pandas as pd

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


def transform_to_2d(X):
    matrix = X.as_matrix()
    matrix.shape = (matrix.shape[0], 1, 7, 3)
    return matrix


def load_dataset(training_fn='numerai_dataset/numerai_training_data.csv',
        validating_fn='numerai_dataset/numerai_tournament_data.csv', as_matrix=False):

    training_data = pd.read_csv(training_fn, header=0)
    # shuffle 
    training_data = training_data.reindex(np.random.permutation(training_data.index))
    validate_data = pd.read_csv(validating_fn, header=0).dropna()

    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]

    x_validate = validate_data[features]
    y_validate = validate_data['target']

    if as_matrix:
        X = transform_to_2d(X)
        x_validate = transform_to_2d(x_validate)

    return X, Y, x_validate, y_validate


def main():
    # prepare the dataset
    X, Y, x_val, y_val = load_dataset(as_matrix=True)

    Y = Y.as_matrix()
    #Y.shape = (Y.shape[0], 1, 1)


    # train the model
    model = conv_model()

    tb_callback = keras.callbacks.TensorBoard(log_dir='./dump', histogram_freq=0,  
          write_graph=True, write_images=True)
    model.fit(X, Y, epochs=32, batch_size=32,  callbacks=[tb_callback])


main()

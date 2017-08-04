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
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(1, 7, 3), data_format='channels_first', name="conv1"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name="conv2"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name="conv3"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), name="conv4"))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["binary_accuracy"])
    return model


def main(grouped=False, visualization=True):
    # prepare the dataset

    if grouped:
        dataset = load_dataset(as_matrix=(1, 7, 3), group_by_era=True, shuffle=True)
        train_set, eras = dataset['training'], dataset['eras']
    X_b, Y_b, x_val, y_val = load_dataset(as_matrix=(1, 7, 3))

    X, Y = train_set[0]
    
    # train the model
    model = conv_model()

    def train_on_era():
        # uses era based batch
        j = 0
        for X, Y in train_set:
            print "training on [ %s ]" % eras[j]
            callbacks = []
            callbacks.append(keras.callbacks.ModelCheckpoint("./checkpoints/weights-{epoch:02d}.hdf5", verbose=0, save_best_only=False, period=100))
            if visualization:
                tb_callback = keras.callbacks.TensorBoard(log_dir='./dump/epoch:%i %s' % (j, eras[j]), histogram_freq=0,
                    write_graph=True, write_grads=True, write_images=True)
                callbacks.append(tb_callback)
            model.fit(X, Y, epochs=1000, batch_size=128,  callbacks=callbacks)
            if j == 90:
                break
            j += 1
        

    def normal_train():
        """
        normal training mode
        """
        if visualization:
            tb_callback = keras.callbacks.TensorBoard(log_dir='./dump/train', histogram_freq=0,  
                write_graph=True, write_grads=True, write_images=True)
        model.fit(X, Y, epochs=700, batch_size=32,  callbacks=[tb_callback])

    # select training method 
    #train_on_era() if grouped else normal_train()

    # evaluate accuracy
    test_loss, test_accuracy = tuple(model.evaluate(X_b, Y_b, verbose=128))
    with file('stats', 'w') as f:
        acc = "test accuracy: " + str(test_accuracy)
        f.write(acc)
        print acc
main(grouped=True)

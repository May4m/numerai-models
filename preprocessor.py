
import numpy as np
import pandas as pd


def transform_to_2d(X, matrix_size):
    """
    transforms a pandas DataFrames from flat rows to a 2D matrix
    """

    matrix = X.as_matrix()
    matrix.shape = [matrix.shape[0]] + list(matrix_size)
    return matrix


def load_dataset(training_fn='numerai_dataset/numerai_training_data.csv',
        validating_fn='numerai_dataset/numerai_tournament_data.csv', as_matrix=None):
    """
    helper function to load the datasets to memory.
    `as_matrix`: turns the features (rows) vector to a matrix
    """

    training_data = pd.read_csv(training_fn, header=0)

    # shuffle: incorrect shuffling protocol
    training_data = training_data.reindex(np.random.permutation(training_data.index))
    validate_data = pd.read_csv(validating_fn, header=0).dropna()

    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]

    x_validate = validate_data[features]
    y_validate = validate_data['target']

    if as_matrix:
        X = transform_to_2d(X, as_matrix)
        x_validate = transform_to_2d(x_validate, as_matrix)
        Y = Y.as_matrix()

    return X, Y, x_validate, y_validate

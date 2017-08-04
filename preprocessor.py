
import numpy as np
import pandas as pd


cache = {}  # temp in mem cache


def transform_to_2d(X, matrix_size=(1, 7, 3)):
    """
    transforms a pandas DataFrames from flat rows to a 2D matrix
    """
    matrix = X.as_matrix()
    matrix.shape = [matrix.shape[0]] + list(matrix_size)
    return matrix


def group_by(df, group, shuffle=False, as_matrix=False):
    """
    creates a list of groups and randomly shuffles the groups
    """
    eras = list(set(df.era))
    np.random.shuffle(eras) if shuffle else None
    groups = [df[df.era == group] for group in eras]
    features = ['feature%i' % i for i in range(1, 22)]

    def _get_vars(frame):
        X = transform_to_2d(frame[features]) if as_matrix else frame[features]
        Y = frame['target'].as_matrix() if as_matrix else frame['target']
        return (X, Y)

    return map(_get_vars, groups), eras


def matrix_form(training_data, features, as_matrix):
    """
    turns to matrix
    """



def load_dataset(training_fn='numerai_dataset/numerai_training_data.csv',
        validating_fn='numerai_dataset/numerai_tournament_data.csv',
        as_matrix=None, group_by_era=False, shuffle=True):
    """
    helper function to load the datasets to memory.
    `as_matrix`: turns the features (rows) vector to a matrix
    """

    # caching section. load_dataset might be called multiple times: #TODO: replace with decorator cache
    if 'tr-data' in cache:
        training_data = cache['tr-data']
        validate_data = cache['vl-data']
    else:
        training_data = cache['tr-data'] = pd.read_csv(training_fn, header=0)
        validate_data = cache['vl-data'] = pd.read_csv(validating_fn, header=0)

    features = ['feature%i' % i for i in range(1, 22)]
    x_validate = validate_data[features]
    y_validate = validate_data['target']
    if as_matrix:
        x_validate = transform_to_2d(x_validate, as_matrix)
        y_validate = y_validate.as_matrix()
    if group_by_era:
        dataset, eras = group_by(training_data, 'era', shuffle=shuffle, as_matrix=as_matrix)
        return {'training': dataset,
                'validation': (x_validate, y_validate),
                'eras': eras}

    X = training_data[features]
    Y = training_data['target']
    if as_matrix:
        X = transform_to_2d(X, as_matrix)
        Y = Y.as_matrix()
    return X, Y, x_validate, y_validate

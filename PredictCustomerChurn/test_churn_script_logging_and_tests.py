"""
Unit tests for churn_library.py
Author: Przemek Sekula
Date: September 2023
"""

import argparse
import pytest
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models

#### FIXTURES ####


@pytest.fixture(scope='module')
def args():
    """ Arguments for the script
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data_path',
                             type=str,
                             default='./data/bank_data.csv',
                             )
    args_parser.add_argument('--eda_path',
                             type=str,
                             default='./test_output',
                             )
    args_parser.add_argument('--model_path',
                             type=str,
                             default='./test_output',
                             )
    args_parser.add_argument('--results_path',
                             type=str,
                             default='./test_output',
                             )
    return args_parser.parse_args()


@pytest.fixture(scope='module')
def encoded_cols():
    """ List of categorical columns that need to be encoded by encoder_helper
    """
    return pytest.data.select_dtypes(include='object').columns.tolist()


#### UNIT TESTS ####

def test_import_data1(args):
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    loaded_df = import_data(args.data_path)

    # We are going to use only first 100 rows, to speed up tests
    pytest.data = loaded_df.head(100)


def test_import_data2():
    """
    test data import - checks if data is a dataframe
    """
    assert isinstance(pytest.data, pd.DataFrame)


def test_import_data3():
    """
    test data import - checks if dataframe is not empty
    """
    assert pytest.data.shape[0] > 0
    assert pytest.data.shape[1] > 0


def test_import_data4():
    """
    test data import - checks if there is label column in data
    """
    assert 'Churn' in pytest.data.columns


def test_eda(args):
    '''
    test perform eda function
    '''
    perform_eda(pytest.data, args.eda_path)


def test_encoder_helper(encoded_cols):
    '''
    test encoder helper
    '''
    data = pytest.data
    encoded_data = encoder_helper(data, encoded_cols)
    assert encoded_data.shape[1] == data.shape[1] + len(encoded_cols)


def test_perform_feature_engineering1():
    '''
    test perform_feature_engineering
    '''
    train_test_data = perform_feature_engineering(pytest.data)

    assert len(train_test_data) == 4
    pytest.train_test_data = train_test_data


def test_perform_feature_engineering2():
    '''
    test perform_feature_engineering - shape check
    '''
    x_train, x_test, y_train, y_test = pytest.train_test_data
    assert x_train.shape[1] == x_test.shape[1]
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_train_models(args):
    '''
    test train_models
    '''
    x_train, x_test, y_train, y_test = pytest.train_test_data

    train_models(
        x_train, x_test, y_train, y_test, args
    )


if __name__ == "__main__":
    print('Just run "pytest"')

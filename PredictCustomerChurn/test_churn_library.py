import pytest
import os
from datetime import datetime, timedelta

import pandas as pd

from churn_library import *


@pytest.fixture
def df_path():
    """ Path to the data file
    """
    return './data/bank_data.csv'

@pytest.fixture
def eda_path():
    """ Path to the folder where the EDA images will be saved
    """
    return './images/eda'

@pytest.fixture
def data(df_path):
    """ Dataframe with the data from the data file
    """
    return import_data(df_path)

@pytest.fixture
def encoded_cols():
    """ List of categorical columns that need to be encoded by encoder_helper
    """
    return [
        'Gender', 'Education_Level', 'Marital_Status', 
        'Income_Category', 'Card_Category'
    ]


def test_import_data(df_path):
    assert(type(import_data(df_path)) == pd.DataFrame)
    
def test_perform_eda(data, eda_path):
    perform_eda(data, eda_path)
    cutoff_time = datetime.now() - timedelta(minutes=1)

    # List all files in the folder and filter by modification time
    recent_files = [
        file for file in os.listdir(eda_path) 
        if os.path.isfile(os.path.join(eda_path, file)) and 
        datetime.fromtimestamp(
            os.path.getmtime(os.path.join(eda_path, file))) > cutoff_time
    ]
    
    assert(len(recent_files) > 0)
    
def test_encoder_helper(data, encoded_cols):
    nr_cols_org = data.shape[1]
    encoded_data = encoder_helper(data, encoded_cols)
    assert(encoded_data.shape[1] == nr_cols_org + len(encoded_cols))
    

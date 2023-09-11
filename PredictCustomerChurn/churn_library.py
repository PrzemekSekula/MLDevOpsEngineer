# library doc string


# import libraries
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''

    if os.path.exists(pth):
        try:
            df = pd.read_csv(pth)
            return df
        except pd.errors.ParserError:
            error_msg = f'File {pth} cannot be parsed.'
            logging.error(error_msg)
            # There is no point in continuing without a data file
            # so we raise an error to stop the program
            raise pd.errors.ParserError(error_msg)
    else:
        error_msg = f'File {pth} does not exist.'
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
            


def perform_eda(df, save_to = './images/eda'):
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe
        save_to: path to save images to [optional argument that could 
            be used for naming variables or index y column]

    output:
        None
    '''
    
    # Histogram of th existing customers
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20,10)) 
    df['Churn'].hist();
    plt.title('Histogram of the Existing Customers')
    plt.savefig(os.path.join(save_to, 'existing_customers.png'))
    
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist();
    plt.title('Histogram of the Customer Age')
    plt.savefig(os.path.join(save_to, 'customer_age.png'))
    
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Histogram of the Marital Status')
    plt.savefig(os.path.join(save_to, 'marital_status.png'))
    
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Histogram of the Total Transaction Count')
    plt.savefig(os.path.join(save_to, 'total_transaction_count.png'))
    
    plt.figure(figsize=(20,10)) 
    sns.heatmap(
        df.corr(numeric_only = True), annot=False, 
        cmap='Dark2_r', linewidths = 2
        )
    plt.title('Heatmap of the Correlation')
    plt.savefig(os.path.join(save_to, 'heatmap.png'))
    
    

def encoder_helper(df, category_lst, response = 'Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the 
    notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be 
        used for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for each categorical column
    '''
    for category in category_lst:
        new_column = f'{category}_{response}'
        try:
            df[new_column] = df[category].map(
                df.groupby(category)[response].mean()
                )
            logging.info(f'SUCCESS: Column {category} has been encoded.')
        except KeyError:
            logging.error(f'Column {category} does not exist.')
            
    return df

def perform_feature_engineering(df, response):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be 
        used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores 
    report as image in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure
    output:
        None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    pass


if __name__ == '__main__':
    bank_data_df = import_data('./data/bank_data.csv')
    perform_eda(bank_data_df)
    bank_data_df = encoder_helper(
        bank_data_df,
        [
            'Gender', 'Education_Level', 'Marital_Status', 
            'Income_Category', 'Card_Category'
        ]
    )

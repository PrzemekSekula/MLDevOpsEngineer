# library doc string
"""
Module for predicting customer churn.
Author: Przemek Sekula
Date: September 2023
Functions:
    import_data - returns a dataframe for the csv found at pth. Adds "Churn"
        column to the df.
    perform_eda - perform eda on df and save the figures
    encoder_helper - helper function to encode each categorical column
    perform_feature_engineering - perform feature engineering on the df
    classification_report_image - produces classification report for training
        and testing results and stores report as image in images folder
    feature_importance_plot - creates and stores the feature importances
    train_models - train, store models and results
"""

# import libraries
import os
import argparse
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report, roc_curve, auc

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

parser = argparse.ArgumentParser(
    description='Predict customer churn based on bank data.')

parser.add_argument('--eda_path',
                    type=str,
                    default='./images/eda',
                    help='Path to the folder where the EDA images are saved'
                    )

parser.add_argument('--results_path',
                    type=str,
                    default='./images/results',
                    help='Path to the folder where the result images are saved'
                    )

parser.add_argument('--model_path',
                    type=str,
                    default='./models',
                    help='Path to the folder where the models are saved'
                    )

parser.add_argument('--data_file',
                    type=str,
                    default='./data/bank_data.csv',
                    help='Path to the data file'
                    )


def import_data(pth):
    '''
    returns dataframe for the csv found at pth. Adds "Churn" column to the df.
    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''

    if os.path.exists(pth):
        try:
            data = pd.read_csv(pth)

            data['Churn'] = data['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1)

            return data
        except pd.errors.ParserError as exc:
            error_msg = f'File {pth} cannot be parsed.'
            logging.error(error_msg)
            # There is no point in continuing without a data file
            # so we raise an error to stop the program
            raise pd.errors.ParserError(error_msg) from exc
        except KeyError as exc:
            error_msg = f'File {pth} does not have the expected columns.'
            logging.error(error_msg)
            raise KeyError(error_msg) from exc
    else:
        error_msg = f'File {pth} does not exist.'
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)


def perform_eda(data, save_folder=None):
    '''
    perform eda on df and save figures to images folder
    input:
        data: pandas dataframe
        save_folder: path to save images to [optional argument that could
            be used for naming variables or index y column]

    output:
        None
    '''

    if save_folder is None:
        args = parser.parse_args()
        save_folder = args.eda_path

    # Histogram of th existing customers
    plt.figure(figsize=(20, 10))
    data['Churn'].hist()
    plt.title('Histogram of the Existing Customers')
    plt.savefig(os.path.join(save_folder, 'existing_customers.png'))

    plt.figure(figsize=(20, 10))
    data['Customer_Age'].hist()
    plt.title('Histogram of the Customer Age')
    plt.savefig(os.path.join(save_folder, 'customer_age.png'))

    plt.figure(figsize=(20, 10))
    data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Histogram of the Marital Status')
    plt.savefig(os.path.join(save_folder, 'marital_status.png'))

    plt.figure(figsize=(20, 10))
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Histogram of the Total Transaction Count')
    plt.savefig(os.path.join(save_folder, 'total_transaction_count.png'))

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        data.corr(numeric_only=True), annot=False,
        cmap='Dark2_r', linewidths=2
    )
    plt.title('Heatmap of the Correlation')
    plt.savefig(os.path.join(save_folder, 'heatmap.png'))

    logging.info('SUCCESS: EDA charts created.')


def encoder_helper(data, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the
    notebook

    input:
        data: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be
        used for naming variables or index y column]

    output:
        df: pandas dataframe with new columns for each categorical column
    '''

    # We do not want to overwrite the original dataframe
    data = data.copy()
    for category in category_lst:
        new_column = f'{category}_{response}'
        try:
            data[new_column] = data[category].map(
                data.groupby(category)[response].mean()
            )
            logging.info('SUCCESS: Column %s has been encoded.', category)
        except KeyError:
            logging.error('Column %s does not exist.', category)
    return data


def perform_feature_engineering(data, response='Churn',
                                test_size=0.3, random_state=42
                                ):
    '''
    input:
        data: pandas dataframe
        response: string of response name [optional argument that could be
        used for naming variables or index y column]

    output:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    # Attrition Flag is the response variable, not a feature, so we do not
    # want to include it in our feature set.
    category_lst = data.drop('Attrition_Flag', axis=1) \
        .select_dtypes(include='object').columns.tolist()

    data = encoder_helper(data, category_lst, response)

    x_train, x_test, y_train, y_test = train_test_split(
        data[keep_cols], data[response],
        test_size=test_size,
        random_state=random_state)

    return x_train, x_test, y_train, y_test


def classification_report_image(model, x_data, y_data, axis, title):
    """Helper function that plots a single image
    input:
        model: Model (LogisticRegression or RandomForestClassifier)
        x_data: Features
        y_data: Labels
        axis: Plot axis
        title: Plot title
    output:
        None
    """
    y_preds = model.predict(x_data)

    report = classification_report(
        y_data, y_preds, output_dict=True)

    df_report = pd.DataFrame(report).transpose()

    df_report.iloc[:-1, :-1].plot(
        kind='bar', ax=axis, title=title)


def feature_importance_plot(model, x_data, save_to=None):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values
        save_to: path to store the figure
    output:
        None
    '''

    if save_to is None:
        args = parser.parse_args()
        save_to = os.path.join(args.results_path, 'feature_importance.png')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)

    plt.figure(figsize=(20, 10))
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.savefig(save_to)
    logging.info('SUCCESS: Feature importance plot has been saved.')


def get_roc_curve_data(model, x_test, y_test):
    """
    Get the data needed to plot the ROC curve
    input:
        model: model object containing predict_proba method
        x_test: X test data
        y_test: y test data
    output:
        fpr: false positive rate
        tpr: true positive rate
        roc_auc: Area Under the Curve.
    """
    preds = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curve(lrc, rfc, x_test, y_test, save_to=None):
    """Plots the ROC curve for the logistic regression and random forest
    input:
        lrc: LogisticRegression object
        rfc: RandomForestClassifier object
        x_test: X test data
        y_test: y test data
    output:
        None
    """

    if save_to is None:
        args = parser.parse_args()
        save_to = os.path.join(args.results_path, 'roc_comparison.png')

    # Get the data needed to plot the ROC curve
    fpr_lr, tpr_lr, roc_auc_lr = get_roc_curve_data(lrc, x_test, y_test)
    fpr_rf, tpr_rf, roc_auc_rf = get_roc_curve_data(rfc, x_test, y_test)

    display_lr = RocCurveDisplay(
        fpr=fpr_lr, tpr=tpr_lr, roc_auc=roc_auc_lr,
        estimator_name='Logistic Regression')
    display_lr.plot(
        color='blue', label=f'Logistic Regression (area = {roc_auc_lr:0.2f})')

    display_rf = RocCurveDisplay(
        fpr=fpr_rf, tpr=tpr_rf, roc_auc=roc_auc_rf,
        estimator_name='Random Forest')
    display_rf.plot(
        ax=plt.gca(), color='red',
        label=f'Random Forest (area = {roc_auc_rf:0.2f})')  # use the same axes

    plt.legend(loc='best')
    plt.title('ROC comparison')
    plt.savefig(save_to)
    logging.info('SUCCESS: ROC comparison plot has been saved.')


def train_models(x_train, x_test, y_train, y_test, args=None):
    '''
    train, store model results: images + scores, and store models
    input:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''

    if args is None:
        args = parser.parse_args()

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    logging.info('SUCCESS: Models trained.')

    lrc.fit(x_train, y_train)
    rfc = cv_rfc.best_estimator_  # for convenience
    joblib.dump(rfc, os.path.join(args.model_path, 'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(args.model_path, 'logistic_model.pkl'))

    plot_roc_curve(
        lrc, rfc, x_test, y_test,
        save_to=os.path.join(args.results_path, 'roc_comparison.png'))

    # Classification Report
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    classification_report_image(
        lrc, x_train, y_train, axes[0, 0], 'Train Logistic Regression')
    classification_report_image(
        rfc, x_train, y_train, axes[0, 1], 'Train Random Forest')
    classification_report_image(
        lrc, x_test, y_test, axes[1, 0], 'Test Logistic Regression')
    classification_report_image(
        rfc, x_test, y_test, axes[1, 1], 'Test Random Forest')
    plt.tight_layout()
    # Save the figure to an images folder
    plt.savefig(os.path.join(args.results_path, 'classification_report.png'))
    plt.close()

    logging.info('SUCCESS: Classification report has been saved.')

    feature_importance_plot(
        rfc, x_train,
        os.path.join(args.results_path, 'feature_importance.png'),
    )


if __name__ == '__main__':
    global_args = parser.parse_args()
    bank_data_df = import_data(global_args.data_file)
    perform_eda(bank_data_df, save_folder=global_args.eda_path)
    xtrain, xtest, ytrain, ytest = perform_feature_engineering(
        bank_data_df)
    train_models(xtrain, xtest, ytrain, ytest, global_args)

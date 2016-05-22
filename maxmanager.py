from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import itertools
import numpy as np
import logging
from copy import deepcopy


# TODO: Add logging statements
# TODO: Test on python 3
# TODO: enable ensembling methods
# TODO: add confusion matrix

class MaxData(object):
    """
    Class to more easily manage data sources that will be used in sklearn classification models
    """

    # class variable that allows us to assign a unique id to each new class instance
    newid = itertools.count().next

    def __init__(self, dataframe, y_var, exclude_columns=None, add_intercept=True, balance_train_classes=False,
                 name='Unnamed MaxData'):
        """
        :param dataframe: DataFrame that is source of data
        :param y_var: str name of column in dataframe that contains data to predict
        :param exclude_columns: list of columns to exclude when processing the data to use against an sklearn model
        :param add_intercept: boolean, whether or not to add an intercept term to the data before running against an
        sklearn estimator
        :param balance_train_classes: boolean or str ('upsample' or 'downsample'). Upsamples or downsamples the train
        data to even unbalanced classes. Default is False (no class balancing).
        :param name: str to more easily identify the MaxData object. Defaults to 'Unnamed MaxData.'
        """

        self.id = MaxData.newid()
        self.orig_dataframe = deepcopy(dataframe)
        self.y_var = y_var
        self.name = name
        self.balanced_dataframe = None
        self.balance_train_classes = balance_train_classes
        if exclude_columns is None:
            exclude_columns = []
        self.exclude_columns = exclude_columns

        df = self.orig_dataframe

        # automatically remove all columns that are not floats or ints since
        # data needs to be numeric to be in the model
        # TODO: allow more types of numerics
        df = df.select_dtypes(include=['float64', 'int64'])

        # exclude the `exclude_columns` from the data we use to make the model
        df = df[[col for col in df.columns if col not in self.exclude_columns]]

        # the instance attribute model_df only contains the data that will be used to make the model
        self.model_df = df

        # construct the feature dataset (data that will be broken down into x_train and x_test)
        self.x_df = df.drop([y_var], 1)
        if add_intercept:
            self.x_df.loc[:, 'intercept'] = 1

        # construct the class dataset (i.e., the data we're trying to predict that will be broken down into y_train
        # and y_test
        self.y_series = df[y_var]

        # create a features instance attribute so we can easily get a model's features
        self.features = self.x_df.columns

        # split data into train and test
        # TODO: enable stratify
        # TODO: enable customization of train/test split params
        # TODO: add validation set
        self.x_train, self.x_test, self.y_train, self.y_test \
            = train_test_split(self.x_df, self.y_series, random_state=4444)

        if balance_train_classes:
            self._balance_train_classes()

        # Set a placeholder for the y_predict. We'll set it once we test the model
        self.y_predict = None

    def _balance_train_classes(self):
        # TODO: test with more than two classes
        # TODO: ability to specify balance % parameter and/or more easily optimize
        """
        Utility function to automatically balance data. Use when data for the classes you're trying to predict are
        unbalanced.
        """
        # join the x_train and y_train data back together so that we can
        # upsample/downsample accordingly
        df = self.x_train.join(self.y_train)
        y_data = self.y_train

        # separate the data by class
        data_type = y_data.dtype
        classes = np.unique(y_data).astype((data_type))
        data = {}
        for c in classes:
            data[str(c)] = df[y_data == c]

        # upsample case to balance classes
        if self.balance_train_classes == 'upsample':
            # get the largest class
            arg_max = str(y_data.value_counts().argmax())
            # max_len = np.round(len(data[arg_max])*0.6)
            # get the size of the largest class dataset so we can upsample smaller classes to that size
            max_len = np.round(len(data[arg_max]))

            # upsample the smaller class datasets, sampling with replacement
            for c in classes:
                if str(c) != str(arg_max):
                    data[str(c)] = data[str(c)].sample(n=max_len, replace=True)

        # downsample to balance classes
        else:
            # get the smallest class
            arg_min = str(y_data.value_counts().argmin())

            # get the size of the smallest class dataset so we can downsample larger classes to that size
            # min_len = np.round(len(data[arg_min])*1.4)
            min_len = np.round(len(data[arg_min]))

            # downsample the larger class datasets
            for c in classes:
                if str(c) != str(arg_min):
                    data[str(c)] = data[str(c)].sample(n=min_len)

        # reconstruct the train dataset with the upsamples/downsampled data
        data_list = []
        for key, val in data.iteritems():
            data_list.append(val)
        new_df = pd.concat(data_list)

        # reset the x_train and y_train to our new balanced datasets
        self.x_train = new_df.drop(self.y_var, 1)
        self.y_train = new_df[self.y_var]


class MaxModel(object):
    """
    Class to more easily manage and use sklearn classification models
    """

    # class variable that allows us to assign a unique id to each new MaxModel instance
    newid = itertools.count().next

    def __init__(self, data, estimator, name=None):

        """
        :param data: instance of MaxData object to run estimator against
        :param estimator: instance of sklearn estimator (e.g., RandomForestClassifier(max_depth=5, n_estimators=10,
        max_features=1)). Currently only tested on KNeighborsClassifier, DecisionTreeClassifier,
        RandomForestClassifier, GaussianNB
        :param name: str to more easily identify the MaxModel
        """
        self.id = MaxModel.newid()
        self.data = data
        self.estimator = estimator
        self.name = name or (get_estimator_name(self.estimator) + ' ' + str(self.data.name))

        # set placeholders for class attributes that will be calculated later
        self.fit = None
        self.y_predict = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        self._result_data = None

        # train and test the model against the provided data
        self.train_model()
        self.test_model()

    def train_model(self):
        """

        :return: None
        """
        self.fit = self.estimator.fit(self.data.x_train, self.data.y_train)

    def test_model(self):
        """
        Test the model and calculate model metrics
        :return: None
        """
        x_test, y_test = self.data.x_test, self.data.y_test
        self.y_predict = self.fit.predict(x_test)
        self.data.y_predict = self.y_predict
        self.accuracy = accuracy_score(y_test, self.y_predict)
        self.precision = precision_score(y_test, self.y_predict)
        self.recall = recall_score(y_test, self.y_predict)
        self.f1 = f1_score(y_test, self.y_predict)
        try:
            y_score = self.fit.decision_function(x_test)
        except:
            y_score = (self.fit.predict_proba(x_test))[:, 1]
        self.fpr, self.tpr, _ = roc_curve(y_test, y_score)
        self.roc_auc = auc(self.fpr, self.tpr)

    @property
    def roc_info(self):
        """
        Utility property to easily get all data needed to construct a ROC curve

        :return: tuple of false_positive_rate, true_positive_rate, ROC_area_under_curve, and MaxModel name
        """
        return self.fpr, self.tpr, self.roc_auc, self.name

    def print_classification_report(self):
        """
        Utility function to print the classification report for a model
        """
        print('Classification Report for: %s', self.name)
        print(classification_report(self.data.y_test, self.y_predict))

    # def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):
    #     # Compute confusion matrix
    #     cm = confusion_matrix(self.y_test, self.y_predict)
    #     plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #     plt.title(title)
    #     plt.colorbar()
    #     tick_marks = np.arange(2)
    #     plt.xticks(tick_marks, iris.target_names, rotation=45)
    #     plt.yticks(tick_marks, iris.target_names)
    #     plt.tight_layout()
    #     plt.ylabel('True label')
    #     plt.xlabel('Predicted label')

    def __label_prediction_results(self, row):
        """
        Utility function to label correct (True) vs incorrect (False) classification predictions
        :param row: row of DataFrame
        :return: boolean: True (correct classification) or False (incorrect classification)
        """

        if row['predict'] == row[self.data.y_var]:
            return True
        else:
            return False

    @property
    def result_data(self):
        """
        Get the full result DataFrame that includes the original test data (x_test and y_test), the predictions (as
        'predict' column),
        and a column for the 'prediction result' (True = correct classification, False = incorrect classification)

        Useful when analyzing where the model makes good/bad predictions

        :return: DataFrame
        """
        x_test, y_test = self.data.x_test, self.data.y_test
        y_predict_df = DataFrame(self.y_predict, columns=['predict'], index=x_test.index)
        self._result_data = pd.concat([x_test, y_test, y_predict_df], axis=1)
        self._result_data.loc[:, 'prediction_result'] = self._result_data.apply(self.__label_prediction_results, axis=1)
        self._result_data = self._result_data.merge(self.data.orig_dataframe)
        return self._result_data


def print_results(max_model_list, include_roc=True):
    """
    Prints the classification reports and ROC curves for a list of MaxModel objects
    :param include_roc: boolean, whether or not to print the ROC Curves. Default is True
    """
    for model in max_model_list:
        model.print_classification_report()

    if include_roc:
        print_roc_curves(max_model_list)


def print_roc_curves(max_model_list):
    """
    Print the ROC Curves for a list of MaxModel objects
    :type max_model_list: list of MaxModel objects
    """
    plt.figure(figsize=(10, 10))
    for model in max_model_list:
        fpr, tpr, roc_auc, name = model.roc_info
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (name, roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def get_estimator_name(estimator):
    return str(estimator).split('(')[0]

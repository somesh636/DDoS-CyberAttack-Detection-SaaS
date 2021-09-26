import h5py
import operator
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.processing import LabelEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from __future__ import print_function

np.random.seed(1337)  # for reproducibility
from sklearn import metrics
from sklearn.base import clone
from keras.datasets import imdb
from keras import callbacks
from sklearn.preprocessing import Normalizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error,
                             mean_absolute_error)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators


class DDoSDetection(object):
    """ Multi-Class DDoS detection using ML algorithms and Neural Networks

    Input Parameters:
    ------------------

    n_hidden_unit: Number of Neurons(Unit) in the hidden layer default
                   dtype: int, default value = 60

    epochs: Number of iterations that the Network iterates over the training dataset
            dtype: int, default value = 200

    eta: Learning rate for the Network
         dtype: float, default value = 0.001

    shuffle: This condition enables the dataset to be shuffled every epoch
             dtype: bool, defalut value = True

    batch_size: The size of each batch used while training
                dtype: int, default value = 1

    RandomState: Immutable value. Mainly used for initializing the weights and biases if value is 1
                 the value of the randomly generated weights becomes fixed.
                 If value is 0 then each time it randomly generated the weights
                 dtype: int, default value = 1

    Return:
    -------
    self: all the parameters accessed by class methods

    """

    def __init__(self, n_hidden_unit=100, epochs=200, eta=0.001, shuffle=False, batch_size=100):
        """ Initialization of Class """

        self.random = np.random.RandomState(1)
        self.n_hidden_unit = n_hidden_unit
        self.epochs = epochs
        self.eta = eta
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rf = RandomForestClassifier()

    def LoadDataset(self):
        """ Load CICDDoS2019 Dataset """
        pass

    def DataPreProcessing(self, data_y):
        """ Label Encoding for dataset """
        le = LabelEncoder()
        data_y_trans = le.fit_transform(data_y)

    def FeatureSelection(self, data_X, data_y):
        """ Feature Selection for DDoS Attacks dataset """
        model = ExtraTreesClassifier(random_state=42)
        new_data = None
        return new_data

    def DDoSRandomForest_Train(self, X_train_std_20, y_train_20):
        """ RandomForest Algorithm Train Function
        Parameters:
        ------------
        X_train_std_20: Validation Dataset
        y_train_20: Label/Target of Validation Dataset

        Returns:
        --------
        None
        """
        self.rf.fit(X_train_std_20, y_train_20)

    def DDoSRandomForest_Predict(self, X_test_std_20):
        """ RandomForest Algorithm Train Function
        Parameters:
        ------------
        X_test_std_20: Test Dataset

        Returns:
        --------
        rf_y_pred: predicted labels for the dataset
        """
        rf_y_pred = self.rf.predict(X_test_std_20)
        return rf_y_pred

    def RoC_Curve(classifier, X_val, y_val, title):
        """ RoC Curve for Classifier
        Parameters:
        ------------
        classifier: Machine Learning Classifier to be Evaluated
        X_val: Validation Dataset
        y_val: Label/Target of Validation Dataset

        Attributes:
        Plots the Graph

        Note: Some part of this Method code is taken
            from Sklearn Website
        """

        lw = 2
        n_classes = 12
        y_test1 = to_categorical(y_val)
        pred_RFC_proba = classifier.predict_proba(X_val)
        y_score = pred_RFC_proba

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test1[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test1.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(20, 10))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        list_class = ['BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NTP', 'DrDoS_NetBIOS', 'DrDoS_SNMP',
                      'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'UDP-lag', 'WebDDoS']
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(list_class[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()



# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, \
    PolynomialFeatures
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import matplotlib.pyplot as plt


class BinaryLabelEncoder(BaseEstimator):

    def __init__(self):
        self.fitted_encoder = {}
        self.binary_columns = ['CODE_GENDER',
                               'NAME_CONTRACT_TYPE',
                               'FLAG_OWN_CAR',
                               'FLAG_OWN_REALTY',
                               'EMERGENCYSTATE_MODE']

    def fit(self, X, y=None):
        for col in self.binary_columns:
            le = LabelEncoder()
            le.fit(X[col][X[col].notna()])
            self.fitted_encoder[col] = le
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.binary_columns:
            le = self.fitted_encoder[col]
            X_.loc[X_[col].notna(), col] = le.transform(
                X_[col][X_[col].notna()]).flatten()
        return X_


class MultipleLabelEncoding(BaseEstimator):

    def __init__(self):
        self.fitted_encoder = {}
        self.multiple_categorical_columns = ['NAME_TYPE_SUITE',
                                             'NAME_INCOME_TYPE',
                                             'NAME_EDUCATION_TYPE',
                                             'NAME_FAMILY_STATUS',
                                             'NAME_HOUSING_TYPE',
                                             'OCCUPATION_TYPE',
                                             'WEEKDAY_APPR_PROCESS_START',
                                             'ORGANIZATION_TYPE',
                                             'FONDKAPREMONT_MODE',
                                             'HOUSETYPE_MODE',
                                             'WALLSMATERIAL_MODE']

    def fit(self, X, y):
        for col in self.multiple_categorical_columns:
            te = TargetEncoder()
            te.fit(X[col][X[col].notna()], y[X[col].notna()])
            self.fitted_encoder[col] = te
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.multiple_categorical_columns:
            te = self.fitted_encoder[col]
            X_.loc[X_[col].notna(), col] = te.transform(
                X_[col][X_[col].notna()])
        return X_


class Imputer(BaseEstimator):

    def __init__(self):
        self.imputer = SimpleImputer()

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_ = X.copy()
        data = self.imputer.transform(X_)
        return pd.DataFrame(data=data,
                            index=list(X_.index),
                            columns=list(X_.columns))


class PolyFeatures(BaseEstimator):
    def __init__(self, degree=3):
        self.degree = degree
        self.columns_to_transform = [
            'EXT_SOURCE_1',
            'EXT_SOURCE_2',
            'EXT_SOURCE_3']
        self.poly_transformer = PolynomialFeatures(self.degree)

    def fit(self, X, y=None):
        features_to_transform = X[self.columns_to_transform]
        self.poly_transformer.fit(features_to_transform)
        return self

    def transform(self, X):
        X_ = X.copy()
        poly_features = self.poly_transformer.transform(
            X_[self.columns_to_transform])
        poly_features_names = self.poly_transformer.get_feature_names()
        X_[poly_features_names] = poly_features
        return X_


class Normalization(BaseEstimator):
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X_ = X.copy()
        X_ = pd.DataFrame(data=self.scaler.transform(
            X_), index=list(X_.index), columns=list(X_.columns))
        return X_


class Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier, threshold=0.150):
        self.threshold = threshold
        self.base_classifier = base_classifier

    def fit(self, X, y):
        self.base_classifier.fit(X, y)
        return self

    def predict(self, X):
        y_pred_proba = self.base_classifier.predict_proba(X)
        y_pred = np.array([1 if proba[1] > self.threshold else 0
                           for proba in y_pred_proba])
        return y_pred

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)





def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "threshold={:.3f}, F3-score={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=60",
        color='black')
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def plot_f3_score(thresholds, fbeta_scores):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.xlabel("Threshold")
    plt.ylabel('F3-score')
    plt.plot(thresholds, fbeta_scores)
    annot_max(np.array(thresholds), np.array(fbeta_scores))
    plt.ylim(0, 0.55)
    plt.show()

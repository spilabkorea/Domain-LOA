"""
The interface for data preprocessing.

Authors:
    LogPAI Team

"""


import pandas as pd
import numpy as np

class Vectorizer(object):

    def fit_transform(self, x_train, window_y_train, y_train):
        self.label_mapping = {eid: idx for idx, eid in enumerate(window_y_train.unique(), 2)}
        self.label_mapping["#OOV"] = 0
        self.label_mapping["#Pad"] = 1
        self.num_labels = len(self.label_mapping)
        return self.transform(x_train, window_y_train, y_train)

    def transform(self, x, window_y, y):
        new_x = x.copy()
        new_x["EventSequence"] = new_x["EventSequence"].map(lambda x: [self.label_mapping.get(item, 0) for item in x])
        window_y = window_y.map(lambda x: self.label_mapping.get(x, 0))
        y = y
        data_dict = {"SessionId": new_x["SessionId"].values, "window_y": window_y.values, "y": y.values, "x": np.array(new_x["EventSequence"].tolist())}
        return data_dict


class Vectorizer_sys(object):

    def fit_transform(self, x_, window_y_):
        self.label_mapping = {eid: idx for idx, eid in enumerate(window_y_.unique(), 2)}
        self.label_mapping["#OOV"] = 0
        self.label_mapping["#Pad"] = 1
        self.num_labels = len(self.label_mapping)
        return self.transform(x_, window_y_)

    def transform(self, x_, window_y_):
        new_x = x_.copy()
        new_x["EventSequence"] = new_x["EventSequence"].map(lambda x: [self.label_mapping.get(item, 0) for item in x])
        window_y_ = window_y_.map(lambda x: self.label_mapping.get(x, 0))
        data_dict = {"SessionId": new_x["SessionId"].values, "window_y": window_y_.values, "x": np.array(new_x["EventSequence"].tolist())}
        return data_dict
        
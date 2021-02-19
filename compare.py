import pandas as pd
from utils import dataloader
from utils.preprocessing import Vectorizer
from utils.preprocessing import Vectorizer_sys
from collections import OrderedDict

from model.Iso_2Auto import *
from model.Iso_Auto import *
from model.LSTM_Autoencoder import *
from model.NN_Autoencoder import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

result = OrderedDict()

# HDFS

train_ratio = 0.8
window_size = 10

struct_log = './data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = './data/HDFS/HDFS_100k.log_anomaly_label.csv' # The anomaly label file

(x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = \
dataloader.load_HDFS(struct_log, label_file=label_file, window='session',
                      window_size=window_size, train_ratio=train_ratio, split_type='uniform')

feature_extractor = Vectorizer()
train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

scaler = MinMaxScaler()
train_dataset['x'] = scaler.fit_transform(train_dataset['x'])
test_dataset['x'] = scaler.transform(test_dataset['x'])

# openstack

train_ratio = 0.8
window_size = 10

train_log = './data/OpenStack/openstack_normal1.log_structured.csv'
test_log1 = './data/OpenStack/openstack_normal2.log_structured.csv'
test_log2 = './data/OpenStack/openstack_abnormal.log_structured.csv'

struct_log_list = [train_log,test_log1,test_log2]
label_list = [0,0,1]

(x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = \
dataloader.load_openstack(struct_log_list, label_list=label_list, window_size=window_size)

feature_extractor = Vectorizer()
train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

scaler = MinMaxScaler()
train_dataset['x'] = scaler.fit_transform(train_dataset['x'])
test_dataset['x'] = scaler.transform(test_dataset['x'])


# train
print('== NN_AUtoencoder Train ==')
(y_, y_pred) = nn_autoencoder(train_dataset,test_dataset)
result['nn_autoencoder'] =  [confusion_matrix(y_, y_pred), accuracy_score(y_, y_pred) , f1_score(y_, y_pred)]

print('== LSTM_AUtoencoder Train ==')
(y_, y_pred) = lstm_autoencoder(train_dataset,test_dataset)
result['lstm_autoencoder'] =  [confusion_matrix(y_, y_pred), accuracy_score(y_, y_pred) , f1_score(y_, y_pred)]

print('== Iso_2Auto Train ==')
(y_, y_pred) = iso_2auto(train_dataset,test_dataset)
result['Iso_2Auto'] =  [confusion_matrix(y_, y_pred), accuracy_score(y_, y_pred) , f1_score(y_, y_pred)]

print('== Iso_Auto Train ==')
(y_, y_pred) = iso_auto(train_dataset,test_dataset)
result['Iso_Auto'] =  [confusion_matrix(y_, y_pred), accuracy_score(y_, y_pred) , f1_score(y_, y_pred)]

print(result)
print(pd.DataFrame(result, index = ['confusion_matrix','accuracy_score','f1_score']))

accuracy_df = pd.DataFrame(result, index = ['confusion_matrix','accuracy_score','f1_score'])
accuracy_df.to_excel('./result/score.xlsx')
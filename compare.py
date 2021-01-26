import pandas as pd
from utils import dataloader
from utils.preprocessing import Vectorizer
from utils.preprocessing import Vectorizer_sys
from collections import OrderedDict

from model.Iso_2Auto import *
from model.Iso_Auto import *
from model.LSTM_Autoencoder import *
from model.NN_Autoencoder import *

result = OrderedDict()
data_set = OrderedDict()

train_ratio = 0.2
window_size = 10


'''
HDFS
'''

struct_log = './data/HDFS_100k.log_structured.csv' # The structured log file
label_file = './data/HDFS_100k.log_anomaly_label.csv' # The anomaly label file

df = pd.read_csv(struct_log)

(x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = \
dataloader.load_HDFS(struct_log, label_file=label_file, window='session',                                                                                                          window_size=window_size, train_ratio=train_ratio, split_type='uniform')

feature_extractor = Vectorizer()
train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)

data_set['HDFS'] = train_dataset['x']/train_dataset['x'].max()

'''
HDFS_Time
'''

struct_log = './data/HDFS_100k.log_structured.csv' # The structured log file
label_file = './data/HDFS_100k.log_anomaly_label.csv' # The anomaly label file

df = pd.read_csv(struct_log)

x_, window_y_ = \
dataloader.load_HDFS(struct_log, window='session',window_size=window_size, train_ratio=train_ratio, split_type='uniform', Time = True)

feature_extractor = Vectorizer_sys()
train_dataset = feature_extractor.fit_transform(x_, window_y_)

data_set['HDFS_Time'] = train_dataset['x']/ train_dataset['x'].max()

'''
Syslog_PID
'''

struct_log = './data/LenMa_syslog.txt_structured.csv' # The structured log file
df = pd.read_csv(struct_log)

x_, window_y_ = \
dataloader.load_sys(struct_log, window='session', window_size=window_size, train_ratio=train_ratio, split_type='uniform')

feature_extractor = Vectorizer_sys()
train_dataset = feature_extractor.fit_transform(x_, window_y_)

data_set['Syslog_PID'] = train_dataset['x']/ train_dataset['x'].max()

'''
Syslog_Time
'''

struct_log = './data/LenMa_syslog.txt_structured.csv' # The structured log file
df = pd.read_csv(struct_log)

x_, window_y_ = \
dataloader.load_sys(struct_log, window='session', window_size=window_size, train_ratio=train_ratio, split_type='uniform',Time =True)

feature_extractor = Vectorizer_sys()
train_dataset = feature_extractor.fit_transform(x_, window_y_)

data_set['Syslog_Time'] = train_dataset['x']/ train_dataset['x'].max()

'''
이벤트_목록(서버로그)
'''

struct_log = './data/이벤트_목록(서버로그).txt_structured.csv' # The structured log file
df = pd.read_csv(struct_log)

x_, window_y_ = \
dataloader.load_sys(struct_log, window='session', window_size=window_size, train_ratio=train_ratio, split_type='uniform',Time =True)

feature_extractor = Vectorizer_sys()
train_dataset = feature_extractor.fit_transform(x_, window_y_)

data_set['이벤트_목록(서버로그)'] = train_dataset['x']/ train_dataset['x'].max()

'''
이벤트_목록_WAS

struct_log = './data/이벤트_목록_waslog.txt_structured.csv' # The structured log file
df = pd.read_csv(struct_log)

x_, window_y_ = \
dataloader.load_sys(struct_log, window='session', window_size=window_size, train_ratio=train_ratio, split_type='uniform',Time =True)

feature_extractor = Vectorizer_sys()
train_dataset = feature_extractor.fit_transform(x_, window_y_)

data_set['이벤트_목록_WAS'] = train_dataset['x']/ train_dataset['x'].max()

'''

'''
train
'''

for data_name, data in data_set.items():
  result[data_name] = []

  result[data_name].append(NN_Autoencoder(data))
  result[data_name].append(LSTM_Autoencoder(data))
  result[data_name].append(Iso_Auto(data))
  result[data_name].append(Iso_2Auto(data))

pd.DataFrame(result, index=['NN_Autoencoder','LSTM_Autoencoder','Iso_Auto','Iso_2Auto'] ).to_csv('result.csv' )
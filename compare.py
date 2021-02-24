import pandas as pd
import argparse

from utils import dataloader
from utils.preprocessing import Vectorizer
from utils.preprocessing import Vectorizer_sys
from collections import OrderedDict

from model.Iso_2Auto import *
from model.Iso_Auto import *
from model.LSTM_Autoencoder import *
from model.NN_Autoencoder import *
from model.SVM_Auto import *
from model.SVM_2Auto import *

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

result = OrderedDict()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'])
  parser.add_argument('-model', type=str, default=None, choices=['nn_autoencoder', 'lstm_autoencoder', 'iso_2auto', 'iso_auto', 'svm_auto', 'svm_2auto'])
  args = parser.parse_args()

  # HDFS
  if args.dataset == 'hd':
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

  # BGL
  elif args.dataset == 'bgl':
    train_ratio = 0.8
    window_size = 10
    struct_log = './data/BGL/BGL_100k.log_structured.csv' # The structured log file

    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = \
    dataloader.load_BGL(struct_log, window_size=window_size, train_ratio=train_ratio, split_type='sequential')

    feature_extractor = Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

    scaler = MinMaxScaler()
    train_dataset['x'] = scaler.fit_transform(train_dataset['x'])
    test_dataset['x'] = scaler.transform(test_dataset['x'])

  # train
  if args.model == None:
    model_list = ['nn_autoencoder', 'lstm_autoencoder', 'iso_2auto', 'iso_auto', 'svm_auto', 'svm_2auto']
  elif args.model != None:
    model_list = [args.model]

  for model_ in model_list:
    print('== {} Train =='.format(model_))
    (y_, y_pred) = locals()[model_](train_dataset,test_dataset)
    result['{}'.format(model_)] =  [confusion_matrix(y_, y_pred), accuracy_score(y_, y_pred) , f1_score(y_, y_pred)]

  accuracy_df = pd.DataFrame(result, index = ['confusion_matrix','accuracy_score','f1_score'])
  print(accuracy_df)
  accuracy_df.to_excel('./result/score.xlsx')
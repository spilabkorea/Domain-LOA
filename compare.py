import time
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
  parser.add_argument('-dataset', type=str, default='hd', choices=['hd', 'bgl'] , help='Choose the Dataset')
  parser.add_argument('-train_ratio' , type=float, default='0.6', help='Train Test Split ratio')
  parser.add_argument('-model', type=str, default=None, choices=['nn_autoencoder', 'lstm_autoencoder', 'iso_2auto', 'iso_auto', 'svm_auto', 'svm_2auto'], help='Select if you only need one model result')
  parser.add_argument('-threshold', type=float, default='0.96', help='Final normal abnormal threshold')
  parser.add_argument('-output_dir', type=str, default='./result/score.xlsx', help='Result excel directory')
  args = parser.parse_args()


  train_ratio = args.train_ratio
  window_size = 10
  # HDFS
  if args.dataset == 'hd':
    # struct_log = './data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
    # label_file = './data/HDFS/HDFS_100k.log_anomaly_label.csv' # The anomaly label file

    struct_log = './data/HDFS/HDFS.log_structured.csv' # The structured log file
    label_file = './data/HDFS/HDFS.log_anomaly_label.csv' # The anomaly label file

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
    # , 'svm_auto', 'svm_2auto'
    model_list = ['nn_autoencoder', 'lstm_autoencoder', 'iso_2auto', 'iso_auto']
  elif args.model != None:
    model_list = [args.model]
  
  for model_ in model_list:
    print('== {} Train =='.format(model_))
    t1 = time.time()
    (y_, y_pred) = locals()[model_](train_dataset,test_dataset, args.threshold)
    t2 = time.time()
    result['{}'.format(model_)] =  [confusion_matrix(y_, y_pred), accuracy_score(y_, y_pred) , f1_score(y_, y_pred), t2-t1]

  accuracy_df = pd.DataFrame(result, index = ['confusion_matrix','accuracy_score','f1_score','time'])
  print(accuracy_df)
  accuracy_df.to_excel(args.output_dir)
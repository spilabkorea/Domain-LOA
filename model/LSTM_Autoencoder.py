import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import  Model, Sequential
from keras.layers import Dense, LSTM , RepeatVector, TimeDistributed, BatchNormalization
from keras.callbacks import EarlyStopping

def lstm_autoencoder(train_dataset, test_dataset):
  x = np.expand_dims(train_dataset['x'],axis=1)

  # LSTM Autoencoder
  model = Sequential()

  # Encoder
  model.add(LSTM(x.shape[-1]//3*2, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(x.shape[-1]//3, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(1, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  # model.add(RepeatVector(10))

  # Decoder
  model.add(LSTM(x.shape[-1]//3, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(x.shape[-1]//3*2, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(TimeDistributed(Dense(x.shape[-1])))

  model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='loss', mode='min')

  # fit
  history = model.fit(x, x,
                      epochs=200, 
                      batch_size=128,
                      callbacks = [early_stopping])
  


  # plot the training losses
  fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
  ax.plot(history.history['loss'], 'b', label='Train', linewidth=2)
  ax.set_title('Model loss', fontsize=16)
  ax.set_ylabel('Loss (mae)')
  ax.set_xlabel('Epoch')
  ax.legend(loc='upper right')
  plt.savefig('./result/lstm_auto_loss.png')

  # plot the loss distribution of the training set
  X_pred = model.predict(x)
  X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
  X_pred = pd.DataFrame(X_pred)

  train_scored = pd.DataFrame()
  Xtrain = train_dataset['x']
  train_scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
  plt.figure(figsize=(16,9), dpi=80)
  plt.title('Loss Distribution', fontsize=16)
  sns.distplot(train_scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
  plt.xlim([0.0,.5])
  plt.savefig('./result/lstm_auto_loss_mae.png')

  # calculate the same metrics for the training set 
  # and merge all data in a single dataframe for plotting
  X_pred_train = model.predict(x)
  X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
  X_pred_train = pd.DataFrame(X_pred_train)

  scored_train = pd.DataFrame()
  scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
  scored_train['Threshold'] = scored_train.quantile(0.9)[0]
  scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
  
  # calculate the loss on the test set
  X_pred = model.predict(np.expand_dims(test_dataset['x'],axis=1))
  X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
  X_pred = pd.DataFrame(X_pred)

  scored = pd.DataFrame()
  Xtest = test_dataset['x']
  scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
  scored['Threshold'] =  scored_train.quantile(0.9)[0]
  scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
  scored.head()

  scored = pd.concat([scored_train, scored])
  ture_ = pd.Series(train_dataset['y']).append(pd.Series(test_dataset['y']))

  return ture_,scored['Anomaly']
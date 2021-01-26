import numpy as np
from keras.models import  Model, Sequential
from keras.layers import Dense, LSTM , RepeatVector, TimeDistributed, BatchNormalization

def LSTM_Autoencoder(train_dataset):
  x = np.expand_dims(train_dataset,axis=1)

  # LSTM Autoencoder
  model = Sequential()

  # Encoder
  model.add(LSTM(8, activation='relu', input_shape=(1,10), return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(4, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(1, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  # model.add(RepeatVector(10))

  # Decoder
  model.add(LSTM(4, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(LSTM(8, activation='relu', return_sequences=True))
  model.add(BatchNormalization())
  model.add(TimeDistributed(Dense(10)))

  model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

  # fit
  history = model.fit(x, x,
                      epochs=100, batch_size=128)
  
  return max(history.history['accuracy'])


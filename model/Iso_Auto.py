import keras
from keras import regularizers
from keras.models import  Model, Sequential
from keras.layers import Dense, Input, BatchNormalization
from sklearn.ensemble import IsolationForest
import numpy as np

def Iso_Auto(train_dataset):
  clf = IsolationForest(random_state = 42, contamination = 0.3)
  clf.fit(train_dataset)
  pred = clf.predict(train_dataset)

  p1 = train_dataset[pred==1]
  p2 = train_dataset[pred==-1]

  # NN Autoencoder model.
  input_dim = p1.shape[1]
  input = Input(shape=(input_dim, ))
  encode = Dense(input_dim//3*2, activation='relu',kernel_regularizer=regularizers.l2(0.01))(input)
  encode = BatchNormalization()(encode)
  encode = Dense(input_dim//3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
  encode = BatchNormalization()(encode)

  decode = Dense(input_dim//3, activation='relu')(encode)
  decode = BatchNormalization()(decode)
  decode = Dense(input_dim//3*2, activation='relu')(decode)
  decode = BatchNormalization()(decode)
  decode = Dense(input_dim, activation='sigmoid')(decode)

  autoencoder = Model(input, decode)

  autoencoder.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

  # Train model.
  history = autoencoder.fit(p1, p1,
          epochs=100,
          batch_size=128,
          shuffle=True
          )

  return max(history.history['accuracy'])
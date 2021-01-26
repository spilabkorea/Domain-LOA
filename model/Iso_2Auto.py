import keras
from keras import regularizers
from keras.models import  Model, Sequential
from keras.layers import Dense, Input, BatchNormalization
from sklearn.ensemble import IsolationForest
import numpy as np

def Iso_2Auto(train_dataset):
  input_dim = train_dataset.shape[1]
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
  history1 = autoencoder.fit(train_dataset, train_dataset,
          epochs=100,
          batch_size=128,
          shuffle=True
          )
  
  f1_pre = autoencoder.predict(train_dataset)

  clf = IsolationForest(random_state = 42, contamination = 0.3)
  clf.fit(f1_pre)
  pred = clf.predict(f1_pre)

  p1 = f1_pre[pred==1]
  p2 = f1_pre[pred==-1]

  
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
  history2 = autoencoder.fit(p1, p1,
          epochs=100,
          batch_size=128,
          shuffle=True
          )

  return (max(history1.history['accuracy']) , max(history2.history['accuracy']))
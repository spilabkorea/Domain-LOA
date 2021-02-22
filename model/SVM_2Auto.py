import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras import regularizers
from keras.models import  Model, Sequential
from keras.layers import Dense, Input, BatchNormalization
from sklearn.svm import OneClassSVM
from keras.callbacks import EarlyStopping

def svm_2auto(train_dataset,test_dataset):
	input_dim = train_dataset['x'].shape[1]
	input = Input(shape=(input_dim, ))
	encode = Dense(input_dim//3*2, activation='relu',kernel_regularizer=regularizers.l2(0.01))(input)
	encode = BatchNormalization()(encode)
	encode = Dense(input_dim//3, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
	encode = BatchNormalization()(encode)
	encode = Dense(1, activation='relu',kernel_regularizer=regularizers.l2(0.01))(encode)
	encode = BatchNormalization()(encode)

	decode = Dense(input_dim//3, activation='relu')(encode)
	decode = BatchNormalization()(decode)
	decode = Dense(input_dim//3*2, activation='relu')(decode)
	decode = BatchNormalization()(decode)
	decode = Dense(input_dim, activation='sigmoid')(decode)

	autoencoder1 = Model(input, decode)
	early_stopping = EarlyStopping(monitor='loss', mode='min')
	autoencoder1.compile(optimizer='adam',
							loss='mse',
							metrics=['accuracy'])

	# Train model.
	history1 = autoencoder1.fit(train_dataset['x'], train_dataset['x'],
					epochs=100,
					batch_size=128,
					shuffle=True,
					callbacks = [early_stopping]
					)

	f1_pre = autoencoder1.predict(train_dataset['x'])

	clf =  OneClassSVM(gamma='auto')
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

	autoencoder2 = Model(input, decode)
	early_stopping = EarlyStopping(monitor='loss', mode='min')
	autoencoder2.compile(optimizer='adam',
							loss='mse',
							metrics=['accuracy'])

	# Train model.
	history2 = autoencoder2.fit(p1, p1,
					epochs=100,
					batch_size=128,
					shuffle=True,
					callbacks = [early_stopping]
					)

	# plot the training losses
	fig, loss_ax = plt.subplots(figsize=(14, 6), dpi=80)
	acc_ax = loss_ax.twinx()
	loss_ax.plot(history1.history['loss'], 'b', label='Train 1 loss', linewidth=2)
	loss_ax.plot(history2.history['loss'], 'r', label='Train 2', linewidth=2)
	loss_ax.set_xlabel('Epoch')
	loss_ax.set_ylabel('Loss (mae)')
	loss_ax.legend(loc='upper right')

	acc_ax.plot(history1.history['accuracy'], 'b', label='Train 1 acc')
	acc_ax.plot(history2.history['accuracy'], 'r', label='Train 2 acc')
	acc_ax.set_ylabel('Accuracy')
	acc_ax.legend(loc='upper right')

	loss_ax.set_title('Model loss,acc', fontsize=16)
	plt.savefig('./result/svm+2auto_loss.png')

	# plot the loss distribution of the training set
	X_pred = autoencoder1.predict(train_dataset['x'])
	X_pred = autoencoder2.predict(X_pred)
	X_pred = pd.DataFrame(X_pred)

	train_scored = pd.DataFrame()
	Xtrain = train_dataset['x']
	train_scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
	plt.figure(figsize=(16,9), dpi=80)
	plt.title('Loss Distribution', fontsize=16)
	sns.distplot(train_scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
	plt.xlim([0.0,.5])
	plt.savefig('./result/svm+2auto_loss_mae.png')

	# calculate the same metrics for the training set 
	# and merge all data in a single dataframe for plotting
	X_pred_train = autoencoder2.predict(autoencoder1.predict(train_dataset['x']))
	X_pred_train = pd.DataFrame(X_pred_train)

	scored_train = pd.DataFrame()
	scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
	scored_train['Threshold'] = scored_train.quantile(0.9)[0]
	scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']

	# calculate the loss on the test set
	X_pred = autoencoder2.predict(autoencoder1.predict(test_dataset['x']))
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
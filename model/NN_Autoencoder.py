from keras.models import  Model
from keras import regularizers
from keras.layers import Dense, Input, BatchNormalization

# Autoencoder model.
def NN_Autoencoder(train_dataset):
        input_dim = train_dataset.shape[1]
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

        autoencoder = Model(input, decode)

        autoencoder.compile(optimizer='adam',
                loss='mse',
                metrics=['accuracy'])


        # Train model.
        history = autoencoder.fit(train_dataset, train_dataset,
                                epochs=100,
                                batch_size=128,
                                shuffle=True
                                )

        return max(history.history['accuracy'])
from tensorflow import keras
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, LSTM, GRU, SimpleRNN, Bidirectional, Dropout

def get_rnn_model(dataset, version):
    if version == "vanilla":
        if dataset == "MITBIH":
            model = keras.Sequential()
            model.add(SimpleRNN(128, input_shape=[None,1], return_sequences=True, dropout=0.3, activation=activations.relu))
            model.add(SimpleRNN(128, return_sequences=True, dropout=0.3, activation=activations.relu))
            model.add(SimpleRNN(128, dropout=0.3, activation=activations.relu))
            model.add(Dense(5, activation=activations.softmax))

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
            
            return model
        else:
            # ptbdb
            model = keras.Sequential()
            model.add(SimpleRNN(128, input_shape=[None,1], return_sequences=True, dropout=0.3, activation=activations.relu))
            model.add(SimpleRNN(128, return_sequences=True, dropout=0.3, activation=activations.relu))
            model.add(SimpleRNN(128, dropout=0.3, activation=activations.relu))
            model.add(Dense(units=1, activation='sigmoid'))

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=losses.binary_crossentropy, metrics=['acc'])
            return model
    else:
        if dataset == "MITBIH":
            model = keras.Sequential()
            model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=[None,1]))
            model.add(Dropout(0.3))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(5, activation=activations.softmax))

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
            return model
        else:
            # ptbdb
            model = keras.Sequential()
            model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=[None,1]))
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(Bidirectional(LSTM(128)))
            model.add(Dropout(0.2))
            model.add(Dense(units=64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=32, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=1, activation='sigmoid'))

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=losses.binary_crossentropy, metrics=['acc'])
            return model
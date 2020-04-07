import pickle
import keras
from keras import regularizers
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, Sequential, load_model
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.constraints import max_norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils

vocab_size = 2000
embedding_size = 16
max_length = 20

def create_base_network(input_dim, is_concat=False):
    sbfl_input = Input(shape=(input_dim,))
    combine_input = sbfl_input
    if is_concat:
        code_input = Input(shape=(None,))
        combine_input = [code_input, sbfl_input]

        output = Embedding(vocab_size, embedding_size, input_length=max_length)(code_input)
        output = LSTM(1, kernel_regularizer=regularizers.l2(0.01))(output)
        output = Activation('sigmoid')(output)
        sbfl_input = concatenate([output, sbfl_input])
    
    output = Dense(16, activation='relu')(sbfl_input)
    output = BatchNormalization()(output)
    output = Dense(8, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dense(1)(output)

    return Model(inputs=combine_input, outputs=output)

def create_meta_network(input_dim, base_network, is_concat=False):
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
    combine_input = [input_a, input_b]
    if is_concat:
        input_a = [Input(shape=(None,)), input_a]
        input_b = [Input(shape=(None,)), input_b]
        combine_input = input_a + input_b

    rel_score = base_network(input_a)
    irr_score = base_network(input_b)
    diff = Subtract()([rel_score, irr_score])
    prob = Activation('sigmoid')(diff)

    return Model(inputs=combine_input, outputs=prob)

def train(n):
    # ochiai is changed to ochiai_x and ochiai_y while merging them.
    combinefl = [technique if technique != 'ochiai' else technique + '_x' for technique in utils.COMBINEFL]
    sbfl = [formula if formula != 'ochiai' else formula + '_y' for formula in utils.FORMULAS]
    keys = sbfl

    tokenizer = pickle.load(open('data/tokenizer.pickle', 'rb'))
    for i in range(n):
        data_dir = 'data/cross_data/{}'.format(i)
        train_df = pd.read_csv('{}/train_pairs.csv'.format(data_dir)).sample(frac=1)
        test_df = pd.read_csv('{}/test_pairs.csv'.format(data_dir)).sample(frac=1)

        train_code1 = pad_sequences(tokenizer.texts_to_sequences(train_df['code_1'].copy()), maxlen=max_length)
        train_code2 = pad_sequences(tokenizer.texts_to_sequences(train_df['code_2'].copy()), maxlen=max_length)
        test_code1 = pad_sequences(tokenizer.texts_to_sequences(test_df['code_1'].copy()), maxlen=max_length)
        test_code2 = pad_sequences(tokenizer.texts_to_sequences(test_df['code_2'].copy()), maxlen=max_length)

        features1 = [feature + '_1' for feature in keys]
        features2 = [feature + '_2' for feature in keys]
        train_features1 = train_df[features1]
        train_features2 = train_df[features2]
        test_features1 = test_df[features1]
        test_features2 = test_df[features2]

        X_train = [train_features1, train_features2]
        X_test = [test_features1, test_features2]

        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        is_concat = False
        if is_concat:
            X_train = [train_code1, train_features1, train_code2, train_features2]
            X_test = [test_code1, test_features1, test_code2, test_features2]
        
        print(test_features1.shape)
        INPUT_DIM = int(len(keys))
        base_network = create_base_network(INPUT_DIM, is_concat)
        model = create_meta_network(INPUT_DIM, base_network, is_concat)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=4096, epochs=4, verbose=1, callbacks=[es])
        base_network.save('{}/ranknet-model.h5'.format(data_dir))

        plt.cla()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig('{}/loss.png'.format(data_dir))

def predict(n):
    # ochiai is changed to ochiai_x and ochiai_y while merging them.
    combinefl = [technique if technique != 'ochiai' else technique + '_x' for technique in utils.COMBINEFL]
    sbfl = [formula if formula != 'ochiai' else formula + '_y' for formula in utils.FORMULAS]
    keys = sbfl

    tokenizer = pickle.load(open('data/tokenizer.pickle', 'rb'))
    for i in range(n):
        data_dir = 'data/cross_data/{}'.format(i)
        test_df = pd.read_csv('{}/test.csv'.format(data_dir))
        
        test_code = pad_sequences(tokenizer.texts_to_sequences(test_df['code'].copy()), maxlen=max_length)
        test_features = test_df[keys].values

        X_test = test_features
        is_concat = False
        if is_concat:
            X_test = [np.array(test_code), test_features]
        
        model = load_model('{}/ranknet-model.h5'.format(data_dir))
        preds = model.predict(X_test)
        np.savetxt('{}/rank-pred.dat'.format(data_dir), preds, newline='\n')

def main():
    stage = 2

    if stage <= 0:
        utils.prepare_data(vocab_size)
    if stage <= 1:
        utils.split_data(n=5)
    if stage <= 2:
        train(n=5)
    if stage <= 3:
        predict(n=5)
    if stage <= 4:
        utils.calc_metric(n=5)

if __name__ == "__main__":
    main()

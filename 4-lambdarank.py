import numpy as np
import pandas as pd
import utils
from lambdarank import LambdaRankNN

def main():
    sbfl = [formula if formula != 'ochiai' else formula + '_y' for formula in utils.FORMULAS]
    keys = utils.SPECTRUM + sbfl + utils.METRICS1
    n = 5

    for i in range(n):
        data_dir = 'data/cross_data/{}'.format(i)
        df = pd.read_csv('{}/train.csv'.format(data_dir))
        
        X = df[keys].values
        y = df['faulty'].values
        qid = df['qid'].values

        ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16, 8), activation=('relu', 'relu'), solver='adam')
        ranker.fit(X, y, qid, epochs=4, batch_size=4096)
        
        X = pd.read_csv('{}/test.csv'.format(data_dir))[keys].values
        y_pred = ranker.predict(X)
        np.savetxt('{}/rank-pred.dat'.format(data_dir), y_pred, newline='\n')

if __name__ == "__main__":
    main()

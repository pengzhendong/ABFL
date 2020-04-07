import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from nltk import FreqDist
from matplotlib.ticker import PercentFormatter

def describe():
    merge = pd.read_csv('data/merge_norm.csv')
    words_cnt = {}
    for line in merge['code'].values:
        for word in line.strip().split():
            if word in words_cnt:
                words_cnt[word] += 1
            else:
                words_cnt[word] = 1
    print('Different tokens: {}'.format(len(words_cnt)))
    sorted_dict = {k: v for k, v in sorted(words_cnt.items(), key=lambda item: item[1], reverse=True)}
    new = open('data/tokens.csv', 'w')
    for key, value in sorted_dict.items():
        new.write('{},{}\n'.format(key, value))

    plt.figure(figsize=(25, 10))
    plt.bar(list(sorted_dict.keys())[:50], list(sorted_dict.values())[:50], color='g')
    pl.xticks(rotation=90)
    plt.savefig('hist.png')
    plt.show()

def analysis():
    # 572039 * 69
    merge = pd.read_csv('data/merge_norm.csv')
    faulty = merge.loc[merge['faulty'] == 1]['dstar2'].to_list()
    nfaulty = merge.loc[merge['faulty'] == 0]['dstar2'].to_list()

    plt.hist(faulty, bins=100, weights=np.ones(len(faulty))/len(faulty), alpha=0.5, label='faulty')
    plt.hist(nfaulty, bins=100, weights=np.ones(len(nfaulty))/len(nfaulty), alpha=0.5, label='non faulty')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.legend(loc='upper right')
    plt.show()

def main():
    # analysis()
    describe()

if __name__ == "__main__":
    main()
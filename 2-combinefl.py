import csv
import os
import random
import subprocess
import utils
import pandas as pd

# ochiai is changed to ochiai_x and ochiai_y while merging them.
combinefl = [technique if technique != 'ochiai' else technique + '_x' for technique in utils.COMBINEFL]
sbfl = [formula if formula != 'ochiai' else formula + '_y' for formula in utils.FORMULAS]
keys =  sbfl

def convert_to_svm_format():
    id2feature = open('data/id-feature.csv', 'w')
    for i, key in enumerate(keys, start=1):
        id2feature.write('{},{}\n'.format(i, key))

    ofile = open('data/l2r_format.dat', 'w')
    data = csv.DictReader(open('data/merge_norm.csv'))
    for row in data:
        features = [row['faulty'], 'qid:' + row['qid']]
        for i, key in enumerate(keys, start=1):
            features.append(str(i) + ':' + str(row[key]))
        ofile.write(' '.join(features) + '\n')

def split_data(cross_project, n):
    random.seed(2019)
    sets = [[] for _ in range(n)]
    if cross_project:
        n = len(utils.PROJECTS)
        qid = 1
        for i, project in enumerate(utils.PROJECTS):
            for _ in range(utils.PROJECTS[project]):
                sets[i].append(str(qid))
                qid += 1
        for l in sets:
            random.shuffle(l) 
    else:
        qid_list = list(range(1, 357 + 1))
        random.shuffle(qid_list)
        cur = 0
        for qid in qid_list:
            sets[cur].append(str(qid))
            cur = (cur + 1) % n

    test_qids = open('data/test-qids.csv', 'w')
    for i in range(n):
        test_qids.write('{},{}\n'.format(i, ' '.join(sets[i])))

        out_dir = 'data/cross_data/{}'.format(i)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print('Writing Train/Test data to {}'.format(i))

        # For ranksvm
        ftrain = open('{}/train.dat'.format(out_dir), 'w')
        ftest = open('{}/test.dat'.format(out_dir), 'w')
        technique = open('{}/rank-pred.dat'.format(out_dir), 'w')
        for line in open('data/l2r_format.dat'):
            items = line.strip().split()
            qid = items[1].split(':')[1]
            if qid in sets[i]:
                ftest.write(line)
                technique.write(items[5].split(':')[1] + '\n')
            else:
                ftrain.write(line)
        print('Test set: {}'.format(sets[i]))

def cross_validation(n):
    processes = []
    for i in range(n):
        # For more feature.
        # 'tools/svm_rank_learn -c 5 {0}/train.dat {0}/svmrank-model.dat &'.format(data_dir)
        # For less feature.
        # 'tools/svm_rank_learn -c 0.01 {0}/train.dat {0}/svmrank-model.dat &'.format(data_dir)
        data_dir = 'data/cross_data/{}'.format(i)
        cmd = 'tools/svm_rank_learn -c 1 {0}/train.dat {0}/svmrank-model.dat'.format(data_dir)
        processes.append(subprocess.Popen(cmd.split(), close_fds=True))
    [p.wait() for p in processes]

    for i in range(n):
        data_dir = 'data/cross_data/{}'.format(i)
        cmd = 'tools/svm_rank_classify {0}/test.dat {0}/svmrank-model.dat {0}/rank-pred.dat'.format(data_dir)
        processes.append(subprocess.Popen(cmd.split(), close_fds=True))
    [p.wait() for p in processes]

def main():
    stage = 1

    if stage <= 0:
        convert_to_svm_format()
    if stage <= 1:
        split_data(cross_project=True, n=5)
    if stage <= 2:
        cross_validation(n=5)
    if stage <= 3:
        utils.calc_metric(n=5)

if __name__ == '__main__':
    main()

import os
import json
import re
import csv
import collections
import tarfile
import random
import math
import pickle
from tqdm import tqdm
from functools import reduce
import operator as op
import pandas as pd
import numpy as np
from numba import jit
from keras.preprocessing.text import Tokenizer

PROJECTS = {'Math': 106, 'Closure': 133, 'Time': 27, 'Chart': 26, 'Lang': 65}
QID = ['qid']
INDEXES = ['project', 'line']
FAULTY = ['faulty']
COMBINEFL = ['stacktrace', 'slicing_count', 'predicateswitching', 'slicing_intersection', 'metallaxis', 'ochiai', 'muse', 'slicing', 'dstar']
FORMULAS = [
    'tarantula', 'ochiai', 'barinel', 'dstar2', 'jaccard', 'er1a',
    'er1b', 'er5a', 'er5b', 'er5c', 'gp2', 'gp3', 'gp13', 'gp19'
]
SPECTRUM = ['ep', 'ef', 'np', 'nf']
CODE = ['code']
METRICS1 = ['level', 'nLines', 'nComments', 'nTokens', 'nChars']
METRICS2 = [
    'nKeyword', 'nModifier', 'nBasicType', 'nSeparator', 'nOperator', 'nAnnotation',
    'nIdentifier', 'nInteger', 'nPoint', 'nBoolean', 'nCharacter', 'nString', 'nNull'
]
METRICS3 = [
    'isAnnotation', 'isNone', 'isAssignment', 'isMethodInvocation', 'isSuperConstructorInvocation',
    'isMemberReference', 'isBinaryOperation' 'isExplicitConstructorInvocation', 'isThis',
    'isSuperMethodInvocation', 'isClassCreator' 'isCast' 'isClassReference', 'isTernaryExpression', 'isLiteral',

    'isStatement', 'isReturnStatement', 'isBreakStatement', 'isStatementExpression', 'isContinueStatement',
    'isThrowStatement', 'isIfStatement', 'isAssertStatement', 'isForStatement', 'isBlockStatement'
]
METRICS = METRICS1 + METRICS2 + METRICS3

def get_src_dir(project, id):
    src_dirs = {'Math': 'src/main/java', 'Closure': 'src', 'Time': 'src/main/java', 'Chart': 'source', 'Lang': 'src/main/java'}
    if (project == 'Math' and int(id) >= 85) or (project == 'Lang' and int(id) >= 36):
        return 'src/java'
    return src_dirs[project]

def stmt_to_line(stmt):
    classname, lineno = stmt.rsplit('#', 1)
    if '$' in classname:
        classname = classname[:classname.find('$')]
    filename = classname.replace('.', '/') + '.java'
    return '{}#{}'.format(filename, lineno)

def parse_test_summary(line, n_elements):
    TestSummary = collections.namedtuple('TestSummary', ('triggering', 'covered_elements'))
    words = line.strip().split(' ')
    coverages, sign = words[:-1], words[-1]
    if len(coverages) != n_elements:
        raise ValueError('Expected {expected} elements in each row, got {actual} in {line!r}'.format(expected=n_elements, actual=len(coverages), line=line))
    return TestSummary(triggering=(sign=='-'), covered_elements=set(i for i in range(len(words)) if words[i]=='1'))

def tally_matrix(matrix_file, n_elements):
    PassFailTally = collections.namedtuple('PassFailTally', ('n_elements', 'passed', 'failed', 'totalpassed', 'totalfailed'))
    summaries = (parse_test_summary(line.decode('utf-8'), n_elements) for line in matrix_file)
    passed = {i: 0 for i in range(n_elements)}
    failed = {i: 0 for i in range(n_elements)}
    totalpassed = 0
    totalfailed = 0
    for summary in summaries:
        if summary.triggering:
            totalfailed += 1
            for element_number in summary.covered_elements:
                failed[element_number] += 1
        else:
            totalpassed += 1
            for element_number in summary.covered_elements:
                passed[element_number] += 1
    return PassFailTally(n_elements, passed, failed, totalpassed, totalfailed)

def nCr(n, r):
    r = min(r, n - r)
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n - r, -1))
    demon = reduce(op.mul, range(1, r + 1))
    return numer // demon

def Einspect(begin, end, mf):
    expected = begin
    m = end - begin + 1
    for k in range(1, m - mf + 1):
        expected += k * nCr(m - k - 1, mf - 1) / nCr(m, mf)
    return expected

def get_Einspect(sorted_lst):
    pos = -1
    for i, f in enumerate(sorted_lst):
        if f['is_fault'] == 1:
            pos = i
            pos_score = f['score']
            break
    if pos == -1:
        return (-1, 0, -1)

    begin = 0
    end = len(sorted_lst) - 1
    for i in range(pos - 1, -1, -1):
        if sorted_lst[i]['score'] != pos_score:
            begin = i + 1
            break
    for i in range(pos + 1, len(sorted_lst)):
        if sorted_lst[i]['score'] != pos_score:
            end = i - 1
            break

    count = 0
    for i in range(begin, end + 1):
        if sorted_lst[i]['is_fault'] == 1:
            count += 1
    return (Einspect(begin + 1, end + 1, count), count, end)

def calc_metric(n):
    qid2lines = {}
    for line in open('data/qid-lines.csv'):
        items = line.strip().split(',')
        qid2lines[items[0]] = items[1]
    
    qid2nfault = {}
    for line in open('data/qid-nfault.csv'):
        items = line.strip().split(',')
        qid2nfault[items[0]] = items[1]

    Epos_list = []
    EXAM_list = []
    AP_list = []
    DCG_list = []
    IDCG_list = []
    for i in range(n):
        data = {}
        curdir = 'data/cross_data/{}'.format(i)
        test_data = open('{}/test.dat'.format(curdir)).readlines()
        pred_data = open('{}/rank-pred.dat'.format(curdir)).readlines()
        for i in range(len(pred_data)):
            test_line = test_data[i]
            pred_line = pred_data[i]
            is_fault = int(test_line.split(' ', 1)[0])
            qid = test_line.split(' ', 2)[1]
            score = float(pred_line)
            if qid not in data:
                data[qid] = []
            item = {'score': score, 'is_fault': is_fault}
            data[qid].append(item)
            
        for key in data.keys():
            sorted_lst = sorted(data[key], key=lambda f: f['score'], reverse=True)
            Einspect, count, end = get_Einspect(sorted_lst)
            Epos_list.append(Einspect)
            # calc EXAM
            qid = key.split(':')[1]
            EXAM = Einspect / float(qid2lines[qid])
            EXAM_list.append(EXAM)
            # calc AP
            if count != 0:
                Einspects = [Einspect for _ in range(count)]
                while end + 1 < len(sorted_lst):
                    cur_Einspect, count, cur_end = get_Einspect(sorted_lst[end + 1:])
                    if cur_Einspect == -1:
                        break
                    Einspects.extend(end + 1 + cur_Einspect for _ in range(count))
                    end += 1 + cur_end
                
                AP = 0
                DCG = []
                for i in range(1, int(qid2lines[qid]) + 1):
                    # 向上取整作为错误所在的具体位置
                    inspects = [math.ceil(Einspect) for Einspect in Einspects]
                    Pi = len([inspect for inspect in inspects if inspect <= i]) / i
                    faulty = 1 if i in inspects else 0
                    AP += Pi * faulty
                    DCG.append(faulty * math.log(2) / math.log(i + 1))
                AP /= float(qid2nfault[qid])
                AP_list.append(AP)
                DCG_list.append(DCG)

                IDCG = []
                for i in range(1, int(qid2nfault[qid]) + 1):
                    IDCG.append(math.log(2) / math.log(i + 1))
                IDCG_list.append(IDCG)
                
    top = []
    top.append(len(list(filter(lambda item: item <= 1 and item > 0, Epos_list))))
    top.append(len(list(filter(lambda item: item <= 3 and item > 0, Epos_list))))
    top.append(len(list(filter(lambda item: item <= 5 and item > 0, Epos_list))))
    top.append(len(list(filter(lambda item: item <= 10 and item > 0, Epos_list))))

    NDCG = [[] for i in range(4)]
    for i in range(len(DCG_list)):
        NDCG[0].append(sum(DCG_list[i][:1]) / sum(IDCG_list[i][:1]))
        NDCG[1].append(sum(DCG_list[i][:3]) / sum(IDCG_list[i][:3]))
        NDCG[2].append(sum(DCG_list[i][:5]) / sum(IDCG_list[i][:5]))
        NDCG[3].append(sum(DCG_list[i][:10]) / sum(IDCG_list[i][:10]))
    for i in range(4):
        NDCG[i] = round(sum(NDCG[i]) / len(NDCG[i]), 4)

    print('ACC@1/3/5/10: {}'.format(top))
    print('NDCG@1/3/5/10: {}'.format(NDCG))
    EXAM_list = [e for e in EXAM_list if e > 0]
    print('EXAM: {}'.format(round(sum(EXAM_list) / len(EXAM_list), 4)))
    print('MAP: {}'.format(round(sum(AP_list) / len(AP_list), 4)))
    
def extract_pairs(labels, qids):
    @jit(nopython=True, cache=True)
    def get_pairs(i, labels, qids):
        pairs = []
        for j in range(i + 1, len(labels)):
            if qids[i] != qids[j]:
                break
            if labels[i] != labels[j]:
                if labels[i] > labels[j]:
                    pairs.append([i, j])
                else:
                    pairs.append([j, i])
        return pairs

    pairs = []
    for i in tqdm(range(0, len(labels))):
        pairs.extend(get_pairs(i, labels, qids))
    print('Found {} pairs.'.format(len(pairs)))
    return pairs

def prepare_data(vocab_size):
    merge_norm = pd.read_csv('data/merge_norm.csv')
    pairs = extract_pairs(merge_norm['faulty'].values, merge_norm['qid'].values)

    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(merge_norm['code'].to_list())
    pickle.dump(tokenizer, open('data/tokenizer.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    merge_norm.drop(['project', 'line', 'faulty'], axis=1, inplace=True)
    colnames = ['label'] + [col + '_1' for col in merge_norm.columns] + [col + '_2' for col in merge_norm.columns]
    ofile = csv.writer(open('data/pairs.csv', 'w'))
    ofile.writerow(colnames)

    data = merge_norm.values.tolist()
    for pair in tqdm(pairs):
        label = random.choice([0, 1])
        aix, bix = (pair[0], pair[1]) if label == 1 else (pair[1], pair[0])
        row = [label]
        row.extend(data[aix])
        row.extend(data[bix])
        ofile.writerow(row)

def split_data(n):
    sets = [[] for _ in range(n)]
    for line in open('data/test-qids.csv'):
        items = line.strip().split(',')
        sets[int(items[0])] = items[1].split()
    
    total = sum(1 for line in open('data/merge_norm.csv')) - 1
    total_pairs = sum(1 for line in open('data/pairs.csv')) - 1
    for i in range(n):
        data_dir = 'data/cross_data/{}'.format(i)

        data = csv.DictReader(open('data/merge_norm.csv'))
        train_csv = csv.DictWriter(open('{}/train.csv'.format(data_dir), 'w'), fieldnames=data.fieldnames)
        test_csv = csv.DictWriter(open('{}/test.csv'.format(data_dir), 'w'), fieldnames=data.fieldnames)
        train_csv.writeheader()
        test_csv.writeheader()
        for row in tqdm(data, total=total):
            if row['qid'] in sets[i]:
                test_csv.writerow(row)
            else:
                train_csv.writerow(row)

        data_pairs = csv.DictReader(open('data/pairs.csv'))
        train_csv = csv.DictWriter(open('{}/train_pairs.csv'.format(data_dir), 'w'), fieldnames=data_pairs.fieldnames)
        test_csv = csv.DictWriter(open('{}/test_pairs.csv'.format(data_dir), 'w'), fieldnames=data_pairs.fieldnames)
        train_csv.writeheader()
        test_csv.writeheader()
        for row in tqdm(data_pairs, total=total_pairs):
            assert row['qid_1'] == row['qid_2']
            if row['qid_1'] in sets[i]:
                test_csv.writerow(row)
            else:
                train_csv.writerow(row)

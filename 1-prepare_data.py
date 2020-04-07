import subprocess
import os
import json
import re
import csv
import collections
import tarfile
import utils
import pandas as pd
from tqdm import tqdm
from formulas import FORMULAS
from javalang.tokenizer import *
from javalang.parser import *

def checkout():
    defects4j = 'tools/defects4j-1.0.1/framework/bin/defects4j'
    processes = []
    for project in utils.PROJECTS:
        for id in range(1, utils.PROJECTS[project] + 1):
            path = 'data/checkout/{}_{}_buggy'.format(project.lower(), id)
            cmd = '{} checkout -p {} -v {}b -w {} &'.format(defects4j, project, id, path)
            processes.append(subprocess.Popen(cmd.split(), close_fds=True))
    [p.wait() for p in processes]

def collect_code():
    ofile = csv.writer(open('data/code.csv', 'w'))
    colnames = utils.INDEXES + utils.CODE
    ofile.writerow(colnames)
    
    release = json.load(open('tools/release.json'.format()))
    for project in tqdm(release):
        lines = release[project]
        [project, id] = re.findall(r'[0-9]+|[a-z]+', project)
        project = project.capitalize()
        src_dir = utils.get_src_dir(project, id)

        stmt2line = {}
        for line in open('tools/source-code-lines/{}-{}b.source-code.lines'.format(project, id)):
            [begin, end] = line.strip().split(':')
            stmt2line[begin] = end
        
        for line in lines:
            line = line.replace('.', '/').replace(':', '.java#')
            [file, begin] = line.split('#')
            if line not in stmt2line:
                end = begin
            else:
                end = stmt2line[line].split('#')[-1]

            java = open('data/checkout/{}_{}_buggy/{}/{}'.format(project.lower(), id, src_dir, file), encoding='latin-1').readlines()
            code = ''.join(java[int(begin) - 1:int(end)])[:-1].replace('\n', '\r\n')
            row = [project.lower() + str(id), line, code]
            ofile.writerow(row)

def collect_spectrum():
    data = 'tools/fault-localization.cs.washington.edu/data'
    ofile = csv.writer(open('data/spectrum.csv', 'w'))
    colnames = utils.INDEXES + utils.SPECTRUM
    ofile.writerow(colnames)
    
    for project in os.listdir(data):
        print('Handling {}...'.format(project))
        project_dir = os.path.join(data, project)
        for id in tqdm(os.listdir(project_dir)):
            gzoltar = os.path.join(project_dir, id, 'gzoltar-files.tar.gz')
            tar = tarfile.open(gzoltar, 'r:gz')
            members = tar.getmembers()
            
            matrix = tar.extractfile(members[1])
            spectra = tar.extractfile(members[2])
            element_names = {i: name.strip().decode('utf-8') for i, name in enumerate(spectra)}
            tally = utils.tally_matrix(matrix, len(element_names))
            totalpassed = tally[3]
            totalfailed = tally[4]

            elements_dict = {}
            for idx in element_names:
                line = utils.stmt_to_line(element_names[idx])
                passed = tally[1][idx]
                failed = tally[2][idx]
                row = [project.lower() + id, line, str(passed), str(failed), str(totalpassed - passed), str(totalfailed - failed)]
                
                # For inner class
                if line not in elements_dict or passed + failed > elements_dict[line]:
                    elements_dict[line] = passed + failed
                    ofile.writerow(row)
    
def json2csv():
    keys = utils.FAULTY + utils.COMBINEFL
    ofile = csv.writer(open('data/release.csv', 'w'))
    ofile.writerow(utils.QID + utils.INDEXES + keys)
    qid2nfault = open('data/qid-nfault.csv', 'w')

    qid = 1
    release = json.load(open('tools/release.json'))
    for project, num in tqdm(utils.PROJECTS.items()):
        for id in range(1, num + 1):
            unique_name = project.lower() + str(id)
            data = release[unique_name]
            nfault = 0
            for line in data:
                row = [qid, unique_name, line.replace('.', '/').replace(':', '.java#')]
                for key in keys:
                    row.append(str(data[line][key]))
                ofile.writerow(row)
                if data[line]['faulty'] == 1:
                    nfault += 1
            qid2nfault.write('{},{}\n'.format(qid, str(nfault)))
            qid += 1

def sbfl():
    ifile = csv.DictReader(open('data/spectrum.csv'))
    ofile = csv.writer(open('data/sbfl.csv', 'w'))
    colnames = utils.INDEXES + utils.FORMULAS
    ofile.writerow(colnames)
    
    for row in ifile:
        for technique in utils.FORMULAS:
            row[technique] = FORMULAS[technique](int(row['ep']), int(row['ef']), int(row['np']), int(row['nf']))
        items = []
        for key in colnames:
            items.append(row[key])
        ofile.writerow(items)

def format_code():
    ifile = csv.DictReader(open('data/code.csv'))
    ofile = csv.writer(open('data/code-info.csv', 'w'))

    colnames = utils.INDEXES + utils.METRICS + utils.CODE
    ofile.writerow(colnames)
    for row in ifile:
        for metric in utils.METRICS:
            row[metric] = 0

        code = row['code'] + '\n'
        tokenizer = JavaTokenizer(code)
        tokens = list(tokenizer.tokenize())
        row['level'] = int((len(code) - len(code.lstrip())) / 2)
        row['nLines'] = code.count('\n')
        row['nComments'] = tokenizer.ncomment
        row['nTokens'] = len(tokens)

        values = []
        for token in tokens:
            type_name = type(token).__name__
            if 'Integer' in type_name:
                type_name = 'Integer'
            elif 'Point' in type_name:
                type_name = 'Point'
            row['n' + type_name] += 1 
            values.append(token.value)
        row['code'] = ' '.join(values)
        row['nChars'] = len(code)

        parser = Parser(tokens)
        if parser.is_annotation():
            row['isAnnotation'] = 1
        try:
            row['is' + type(parser.parse_expression()).__name__] = 1
        except:
            pass
        try:
            row['is' + type(parser.parse_statement()).__name__] = 1
        except:
            pass
        
        items = []
        for key in colnames:
            items.append(row[key])
        ofile.writerow(items)

def merge():
    release = pd.read_csv('data/release.csv')
    spectrum = pd.read_csv('data/spectrum.csv')
    sbfl = pd.read_csv('data/sbfl.csv')
    code = pd.read_csv('data/code-info.csv')
    merge = release.merge(spectrum, on=utils.INDEXES).merge(sbfl, on=utils.INDEXES).merge(code, on=utils.INDEXES)
    merge.to_csv('data/merge.csv', index=False)

    ignore_cols = set(utils.QID + utils.INDEXES + utils.FAULTY + utils.COMBINEFL + utils.CODE)
    for col in set(merge.columns) - ignore_cols:
        merge[col] = merge.groupby('qid')[col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    merge.fillna(0, inplace=True)
    merge.to_csv('data/merge_norm.csv', index=False)
    
def main():
    stage = 6

    if stage <= 0:
        checkout()
    if stage <= 1:
        collect_code()
    if stage <= 2:
        collect_spectrum()
    if stage <= 3:
        json2csv()
    if stage <= 4:
        sbfl()
    if stage <= 5:
        format_code()
    if stage <= 6:
        merge()
    
if __name__ == "__main__":
    main()

#!/bin/bash

# Checkout tool
defects4j=defects4j-1.0.1
if [ ! -f $defects4j.tar.gz ]; then
    wget https://github.com/rjust/defects4j/archive/v1.0.1.tar.gz -O $defects4j.tar.gz &&
fi
tar -zxvf $defects4j.tar.gz &&
sh $defects4j/init.sh

# CombineFL for labels
combinefl=CombineFL-v0.1
if [ ! -f $combinefl.tar.gz ]; then
    wget https://combinefl.github.io/downloads/CombineFL-v0.1.tar.gz -O $combinefl.tar.gz &&
fi
tar -zxvf CombineFL-v0.1.tar.gz &&
cp combinefl/data/release.json ./ &&
cp combinefl/data/qid-lines.csv ../data/ &&
rm -rf *ombine*

# Statements to line
source_code_lines=source-code-lines
if [ ! -f $source_code_lines.tar.gz ]; then
    wget https://bitbucket.org/rjust/fault-localization-data/src/master/analysis/pipeline-scripts/source-code-lines.tar.gz &&
fi
tar -zxvf source-code-lines.tar.gz

# Spectrum
wget --recursive --no-parent --accept gzoltar-files.tar.gz http://fault-localization.cs.washington.edu/data

# SVMRank
if [ ! -f svm_rank.tar.gz ]; then
    wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz
fi
tar -zxvf svm_rank.tar.gz && 
make &&
rm -rf svm_light svm_struct LICENSE.txt Makefile

# Autoencoder
git clone https://github.com/erickrf/autoencoder.git
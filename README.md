## ABFL: An Autoencoder Based Practical Approach for Software Fault Localization

``` bash
cd tools && ./install.sh
```

### javalang

Add `ncomment` field in `tokenizer.py` to count the num of comments

``` python
class JavaTokenizer(object):
    self.ncomment = 0
    ...
    def read_comment(self):
        self.ncomment += 1
    ...
```

### Autoencoder

After running `1-prepare_data.py` to generate code tokens, prepare sequence data and vocabulary for autoencoder.

``` bash
$ python tools/autoencoder/src/prepare-data.py data/code.dat data --max-length 30 --min-freq 100 --valid 0.2
```

### LambdaRank

``` bash
$ git clone https://github.com/liyinxiao/LambdaRankNN.git
```
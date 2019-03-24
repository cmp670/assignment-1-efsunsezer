# Simple pure Python 3 program for n-gram language modelling

### Usage

1. Place source dataset 
2. Run the single python script n-gram_ef.py

### What does this script do?


1. Takes the dataset, performs cleaning, tokenization on it.

2. Creates vocabulary,

3. Builds unigram, bigram and trigram language models.

4. Applies add one smoothing and linear interpolation

5. Estimates MLE probability

6. Generate sentences


###Note about sentence generation part

If start_seq is not given, code chooses a random one. 

### Dependencies
  * [Python 3]
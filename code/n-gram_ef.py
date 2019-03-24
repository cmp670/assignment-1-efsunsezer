from __future__ import division
import numpy as np
import random, re
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
import sklearn 
import string
from collections import deque
from itertools import islice
import collections
import math

# GLOABLS
#STOP_token = '_STOP_'
UNK_token = '_UNK_' 
# token for ngram that appear in dev and test but not in training
UNSEEN_NGRAM = '_UNSEEN_'
SEP = " " 
#unk_threshold = 1
# total counts of words in training data. contrain duplicated word
total_words_len = 0
ADD_K_SMOOTHING = 'add_k_smoothing'
LINER_INT = 'liner interpolation'
NO_SMOOTHING = 'no smoothing'
# training tokens
replaced_tokens_train = list()
vocabulary = set()

class Ngrams:
    
    def __init__(self, ngram_order):  
        #Constructor method
        self.ngram_order =ngram_order
                
    ''' calculate MLE Probablity of unigram
        input word-freq dict for training data, which is Vocaborary
        this function will run even n specified by the shell is not 1
    '''
    def unigrams_prob(self,uni_count_dict):
        # probability dict {word:prob}
        prob_dict = uni_count_dict
        #print vocabulary
        items = prob_dict.items() 
        for word, count in items:
            prob_dict[word] = float(count) / float(total_words_len)
        return prob_dict
          
    '''
    calculate MLE probability of ngram, n>=2
    : param: n: count dict of ngram,start from bigram
    : param: input untokened train texts with STOP sign
    '''
    def ngram_prob(self,n,tokens, unigram_count):
        #print('------start ngram_prob---------------')
        # generate {ngrams:count} from training data
        ngram_list = list(self.ngrams_gen(tokens,n))
        
        ngram_count_pairs = self.word_freq(ngram_list)
        prob_dict = ngram_count_pairs
        if(n == 2):
            items = prob_dict.items()     
            uni_count = unigram_count
            # current probablity and word, in case n = 2, input is bigram words:count dict
            # input {a,b}: count, continue to get {a}: count
            for words, count in items:
                # extract the first item in bigram. 
                prior_word = words[0]   
                # get the count from {unigram: count} generated before       
                cnt_prior = uni_count[prior_word]    
                #print(prior_word,words,cnt_prior,count)  
                # q(w/v) = c(v,w)/c(v)      
                prob_dict[words] = count / cnt_prior
                #print(count,cnt_prior)
            # this should save as global for later use as bigram_prob_dict
            return prob_dict
        if(n > 2):
            items = prob_dict.items() 
            # get {n-1gram:count} pairs
            priorgram_list = list(self.ngrams_gen(tokens,n-1))
            priorgram_count_pairs = self.word_freq(priorgram_list)
            #-----------need to discard first few items--------
            for words, count in items:
                prior_word = words[:n-1]
                cnt_prior = priorgram_count_pairs[prior_word]
                #print(prior_word,words,cnt_prior,count)
                prob_dict[words] = count / cnt_prior
            return prob_dict

    #def sprob(self,):
    
    def ngrams_gen(self,tokens, n):
        ngrams_tp = tuple()
        text = ' '.join(tokens)
        #text = text.replace(STOP_token,STOP_token+'\n')
        sentences = set([w for w in text.splitlines()])
        for word in sentences:
            it = iter(word.split())
            window = deque(islice(it, n), maxlen=n)
            yield tuple(window)       
            for item in it:
                window.append(item)
                yield tuple(window)
           
        ngrams_tp += tuple(window)
        yield ngrams_tp
    
    def ppl(self,test_text,n,prob_dict,smooth_type):  #Returns the perplexity of the given list of sentences

        return math.pow(2.0, self.entropy(test_text,n,prob_dict,smooth_type))
        
        
    ''' calculate entropy given a test/dev text
    # input text should be processed propriately
    # N = 1,2,3
    # smooth_type used to deal with unseen word in different smoothing method
    '''
    def entropy(self,test_test,n,prob_dict,smooth_type):
        entr = 0.0
        text = test_test
        tokens = nltk.word_tokenize(text)
        # number of words in text
        text_len = len(tokens)   
        global vocabulary
        
        sentences = set([s for s in text.splitlines()])
        # number of sentences
        sent_num = len(sentences)
        voc_set = set(prob_dict.keys())
       
        if (n ==1):      
            
            for sent in sentences:
                sent_temp = nltk.word_tokenize(sent)
                for word in sent_temp:
                    if word not in voc_set:
                        entr += self.logprob(UNK_token, prob_dict)
                    else:
                        entr += self.logprob(word, prob_dict)
        if(n > 1):   
            #ngram_prob_dict = ngram_prob(n,train_cut)
            for sent in sentences:
                # generate ngram for single sentence test data
                ngram_tmp = tuple(self.ngrams_gen(nltk.word_tokenize(sent), n))
                # iterate ngram in one sentence, skip first n-1 items
                for i in range(n - 1, len(list(ngram_tmp))):
                    #print i, ngram_tmp[i]
                    if ngram_tmp[i] not in voc_set:
                        if(smooth_type==NO_SMOOTHING):
                            entr += -math.log(0, 2)
                        if(smooth_type==ADD_K_SMOOTHING):
                            entr += self.logprob(UNSEEN_NGRAM, prob_dict)
                            
                    else:
                        entr += self.logprob(ngram_tmp[i], prob_dict)
        return entr / float(text_len - (n - 1)*sent_num)
        
    def logprob(self,word,prob_dict):
            prob_dict = prob_dict
            return -math.log(prob_dict[word], 2)
        
    
    #Samples a word from the conditional distribution of given context. 
    def next_word(self,text, N, counts):
        """ Outputs the next word to add by using most recent tokens """
    
        token_seq = SEP.join(text.split()[-(N-1):]);
        
        try:
            choices = counts[token_seq].items();
        except KeyError:
            #print("word does not found. Sentence is not generated.")
            t='\n'
            return (t)
    
        # make a weighted choice for the next_token
        # [see http://stackoverflow.com/a/3679747/2023516]
        total = 0
        for choice, weight in choices:
            total += weight;
 
        r = random.uniform(0, total)
        upto = 0
        for choice, weight in choices:
            upto += weight;
            if upto > r: return choice
        assert False      
    def ngram_freqs(self,ngram_list):
        """ Builds dict of TOKEN_SEQUENCEs and NEXT_TOKEN frequencies """
    
        ### has form TOKEN_SEQUENCE : DICT OF { NEXT_TOKEN : COUNT }
        ###      e.g        "a b c" : {"d" : 4, "e" : 2, "f" : 6 }
        counts = {}
    
        # Using example of ngram "a b c e" ...
        for ngram in ngram_list:
            token_seq  = SEP.join(ngram[:-1])   # "a b c"
            last_token = ngram[-1]              # "e"
    
            # create empty {NEXT_TOKEN : COUNT} dict if token_seq not seen before
            if token_seq not in counts:
                counts[token_seq] = {};
    
            # initialize count for newly seen next_tokens
            if last_token not in counts[token_seq]:
                counts[token_seq][last_token] = 0;
    
            counts[token_seq][last_token] += 1;
    
        return counts;
                      # should not reach here
                      
    def next(self,N ,ngram_list,start_seq=None):
        
        """ Generate a random sentence based on input text corpus """
        sentence_length=10
       
        counts = self.ngram_freqs(ngram_list)
    
        if start_seq is None: start_seq = random.choice(counts.keys());
        #rand_text = start_seq.lower();
        rand_text = start_seq
        sentences = 0;
        while sentences < sentence_length:
            rand_text += SEP + self.next_word(rand_text, N, counts);
            
            if rand_text.endswith(('.','!', '?','\n')):
                sentences += 1 
            else :
                sentences += 0 
        return rand_text;
    '''
    linear interpolation trigram, use ngrams_prob()
    output perplexity directly
    '''
        
    def add_k_smoothing(self,n,tokens, unigram_count,k,V):
        # generate {ngrams:count} from training data
        # print('------start add_k_smoothing---------------')
        if(n == 1):
            prob_dict = unigram_count
        else:
            ngram_list = list(self.ngrams_gen(tokens,n))
            ngram_count_pairs =self.word_freq(ngram_list)
            prob_dict = ngram_count_pairs
        if (n == 1):
            
            items = prob_dict.items() 
            for word, count in items:
                prob_dict[word] = (float(count)+k)/ (float(total_words_len)+V*k)
            return prob_dict
    
        #print prob_dict
        if(n == 2):
            items = prob_dict.items() 
            # should fix this duplicated thing
            uni_count = unigram_count
            #print len(uni_count)
            # current probablity and word, in case n = 2, input is bigram words:count dict
            # input {a,b}: count, continue to get {a}: count
            for words, count in items:
                # extract the first item in bigram. 
                prior_word = words[0]   
                # get the count from {unigram: count} generated before       
                cnt_prior = uni_count[prior_word]    
                #print(prior_word,words,cnt_prior,count)  
                # q(w/v) = c(v,w)/c(v)      
                prob_dict[words] = (count+k)/ (cnt_prior + k*V)
                #print(count,cnt_prior)
            # dealing with unseen ngram that might appear in test data
            prob_dict[UNSEEN_NGRAM]= 1/V
            # this should save as global for later use as bigram_prob_dict
            return prob_dict
        if(n > 2):
            items = prob_dict.items() 
            # get {n-1gram:count} pairs
            priorgram_list = list(self.ngrams_gen(tokens,n-1))
            priorgram_count_pairs =self. word_freq(priorgram_list)
        
            for words, count in items:
                prior_word = words[:n-1]
                cnt_prior = priorgram_count_pairs[prior_word]
                #print(prior_word,words,cnt_prior,count)
                prob_dict[words] = (count+k)/ (cnt_prior + k*V)
            prob_dict[UNSEEN_NGRAM]= 1/V
            return prob_dict

    def linear_interpolation(self,dev_text, unigrams_prob_dict,bigram_prob_dict,trigram_prob_dict,
    la1,la2,la3):
        entr = 0.0
        perplexity = 0.0
        #global vocabulary
        text = dev_text
        tokens = nltk.word_tokenize(text)
        # number of words in text
        text_len = len(tokens)   
        sentences = set([s for s in text.splitlines()])
        # number of sentences
        sent_num = len(sentences)
        n = 3
        new_prob_dict = {}
        bigram_p_dict = bigram_prob_dict.copy()
        trigram_p_dict = trigram_prob_dict.copy()
        unigram_p_dict = unigrams_prob_dict.copy()
    
        bi_keys_set = set(bigram_p_dict.keys())
        uni_keys_set = set(unigram_p_dict.keys())
        tri_keys_set = set(trigram_p_dict.keys())
    
    
        for sent in sentences:
            # generate trigram for dev/test data
            ngram_tmp = tuple(self.ngrams_gen(tokenize_unigram(sent), n))
                # iterate ngram in one sentence, skip first n-1 items
            for i in range(n - 1, len(list(ngram_tmp))):
                    #print i, ngram_tmp[i]
                    words = ngram_tmp[i]
                    bi_word = words[1:]
                    uniword = words[2]
                    
                    # deal with unseen words
                    if uniword not in uni_keys_set:
                        unigram_p_dict[uniword] = unigram_p_dict[UNK_token]
                    if ngram_tmp[i] not in tri_keys_set:
                        trigram_p_dict[ngram_tmp[i]] = 0
                    if bi_word not in bi_keys_set:
                        bigram_p_dict[bi_word] = 0
                    new_prob_dict[ngram_tmp[i]] = float(la1) * trigram_p_dict[ngram_tmp[i]] +float(la2) * bigram_p_dict[bi_word] + float(la3) * unigram_p_dict[uniword]
                    # get entropy
                    entr += self.logprob(ngram_tmp[i], new_prob_dict)
        # get perplexity
        return math.pow(2.0, (entr / float(text_len - (n - 1)*sent_num)))
    
    ''' 
    only for n=2,3 or more, generate {words:count}
    usually take training ngram tokens such as {a,b,c} as input, 
    generate ngram count with UNK
    when input test/dev data, it is used for error analysis
     
    ''' 
    def word_freq(self,tokens):
        ngram_freq = {}
        # initial work-count dict population
        for token in tokens:       
            ngram_freq[token] = ngram_freq.get(token,0) + 1
        return ngram_freq
        

 
def text_preprocessing(filename):
    s = open(filename, 'r',encoding='utf8').read()
    s = re.sub('[()]', r'', s)                              # remove certain punctuation chars
    s = re.sub('([.-])+', r'\1', s)                         # collapse multiples of certain chars
    s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)  # pad sentence punctuation chars with whitespace
    s = ' '.join(s.split()).lower()                         # remove extra whitespace (incl. newlines)
    return s;
 

'''tokenize processed text'''
def tokenize_unigram(text):
    return nltk.word_tokenize(text)


'''get Vocabulary from training set. including UNK
   :param: input tokenized training text
   :output: unigrams {unigrams:count}
'''
def unigram_V(training_tokens,unk_threshold):
        # this is the total length/num of tokens in training data
        global total_words_len
        global replaced_tokens_train 
        total_words_len = len(training_tokens)
        # initialize word_count pairs
        unigram_V = {}
        unigram_V[UNK_token] = 0
        # initial work-count dict population
        for token in training_tokens:
            unigram_V[token]= unigram_V.get(token,0) + 1
        # re-assign UNK
        unk_words = set()
        items = unigram_V.items()
        for word, count in items:
            # treat low freq word as UNK
            if count <= unk_threshold:
                unk_words.add(word)
                unigram_V[UNK_token] += count
           
        #unk_words.discard(STOP_token)
        unk_words.discard(UNK_token)

        for word in unk_words:
            del unigram_V[word]

        replaced_tokens_train = training_tokens
        for idx, token in enumerate(replaced_tokens_train):            
            if token in unk_words:                
                replaced_tokens_train[idx] = UNK_token
#               modify tuple to contain UNK
        return unigram_V
    
'''
once vocabulary obtained
replace unk in training and test data 
param: input unigram tokens
'''
def replace_UNK(tokens,vocabulary):
        
    for idx, token in enumerate(tokens):            
        if token not in vocabulary:
#               modify tuple to contain UNK
            token_ls = list(tokens)
            token_ls[idx] = UNK_token
            tokens = tuple(token_ls)  
    return tokens


'''ngram generator, n>1
   : param: input tokened texts with STOP sign, and UNK replaced
            could be either training data or test data or sentences
        
'''

    
 
            
##
##Evaluate the (negative) log probability of this word in this context.
##:param word: the word to get the probability of
#:param prob_dict: the context the word is in
##

'''
 input n, test and training, right now no smoothing 
 output perplexity
'''
def main():  
  
    unk_threshold = 1
    # process train and dev/test data
    
    train_text = text_preprocessing("./brown.train.txt")
    dev_text = text_preprocessing("./brown.dev.txt")
    test_text=text_preprocessing("./brown.test.txt")
    
    
    train_token = nltk.word_tokenize(train_text)
    print('tokenization is finished')
    
    ngram_obj =Ngrams(1)  
    unigram_count = unigram_V(train_token,unk_threshold)
    # a list of vocabulary in unigrams
    vocabulary = set(unigram_count.keys())  
    # generate unigram probablity dict
    uni_prob_dict = {}
    uni_prob_dict = unigram_count.copy()
    
    
    unigrams_prob_dict = ngram_obj.unigrams_prob(uni_prob_dict)
    V = len(vocabulary)
    print("Vocabulary lenth",V)
    print('total_words_len',total_words_len)
    print("training unigram finished")
    
    #get perplextity for unigram for dev data
    print('development set perplexity for unsmoothed unigram:')
    print(ngram_obj.ppl(dev_text,1,unigrams_prob_dict,NO_SMOOTHING))
    print('\n')
    
    print('development set perplexity for add 1 unigram:')
    uni_addk_prob_dict = ngram_obj.add_k_smoothing(1,replaced_tokens_train,unigram_count, 1,V)
    print(ngram_obj.ppl(dev_text,1,uni_addk_prob_dict,ADD_K_SMOOTHING))
    
    print('test set perplexity for add 1 unigram:')
    uni_addk_prob_dict = ngram_obj.add_k_smoothing(1,replaced_tokens_train,unigram_count, 1,V)
    print(ngram_obj.ppl(test_text,1,uni_addk_prob_dict,ADD_K_SMOOTHING))
    
    
    #get perplextity for unigram for dev data
    print('test set perplexity for unsmoothed unigram:')
    print(ngram_obj.ppl(test_text,1,unigrams_prob_dict,NO_SMOOTHING))
    print('\n')
    
    # generate bigram probability dict
    bigram_prob_dict =ngram_obj.ngram_prob(2,replaced_tokens_train, unigram_count)
    print("training bigram finished")
    # generate trigram probability dict
    trigram_prob_dict = ngram_obj.ngram_prob(3,replaced_tokens_train, unigram_count)
    print("training trigram finished")
    
    # --------------------------Random sentence generation----------------------------
    print("Random sentence generation")
    unigram_list=list(unigrams_prob_dict.keys())
    #dprint("Unigram:  Generate a random sentence based on input text corpus")
    #uni_sentence=ngram_obj.next(1,unigram_list, start_seq=None)
    #print(uni_sentence)
    print("Bigram:  Generate a random sentence based on input text corpus")
    bigram_list=list(bigram_prob_dict.keys())
    bi_sentence=ngram_obj.next(2,bigram_list, start_seq="I believe")
    print(bi_sentence)
    print("Trigram:  Generate a random sentence based on input text corpus")
    trigram_list=list(trigram_prob_dict.keys())
    tri_sentence=ngram_obj.next(3,trigram_list, start_seq="The Belgians would")
    print(tri_sentence)
    
    # --------------------------uncomment this block if only wanna run on test-------
    # on dev data
    # add k smoothing 
    """
    print('perplexity for add-k-smoothing on development data:')
    k_ls = (0.0000001,0.000001,0.00001,0.0001,0.01,0.1,1)
    for k in k_ls:
        print('k=:',k)
        print('perplexity of add k unigram:')
        uni_addk_prob_dict = ngram_obj.add_k_smoothing(1,replaced_tokens_train,unigram_count, k,V)
        print(ngram_obj.ppl(dev_text,1,uni_addk_prob_dict,ADD_K_SMOOTHING))

        print('perplexity for add k bigram:')
        bi_addk_prob_dict = ngram_obj.add_k_smoothing(2,replaced_tokens_train,unigram_count, k,V)
        print(ngram_obj.ppl(dev_text,2,bi_addk_prob_dict,ADD_K_SMOOTHING))

        print('perplexity for add k trigram:')
        tri_addk_prob_dict = ngram_obj.add_k_smoothing(3,replaced_tokens_train,unigram_count, k,V)
        print(ngram_obj.ppl(dev_text,3,tri_addk_prob_dict,ADD_K_SMOOTHING))
        print('\n')
    print('------end add_k_smoothing---------------')
    print('\n')
    """   
    """  
   # linear interpolation
    print('------liner interpolation on development data-----')

    la_ls = [(0.001,0.006,0.009),(0.1,0.2,0.5),(0.7,0.4,0.3),(0.08,0.009,0.006)]
    perplexity_all = [] # empty list
    for lamda in la_ls:
        print('lamda1,lamda2,lamda3:',lamda[0],lamda[1],lamda[2])
        perplexity_li = ngram_obj.linear_interpolation(dev_text, unigrams_prob_dict,bigram_prob_dict,
    trigram_prob_dict,lamda[0],lamda[1],lamda[2])
        print('linear interpolation perplexity:')
        print(perplexity_li)
        perplexity_all.append(perplexity_li)
    print('\n')
    min_value = min(perplexity_all)
    min_index = perplexity_all.index(min_value)"""
    
    #------------------------end of dev data --------------------------------------------

    #---------------------------------------------------------------------------#
    #-----------------uncomment the following chunk to run test data------------#
    # run on test data
  
    #add k smoothing on test
    # print('perplexity for add-k-smoothing on test data:')
    # k = 0.0000001
    # print('k=:',k)
    
    # print('perplexity for add k trigram:')
    # tri_addk_prob_dict = add_k_smoothing(3,replaced_tokens_train,unigram_count, k,V)
    # print(perplexity(dev_text,3,tri_addk_prob_dict,ADD_K_SMOOTHING))
    # print('\n')
 
    # print('------end add_k_smoothing---------------')
    # print('\n')
   
   
   #linear interpolation
    # print('------liner interpolation on test data-----')
    # lamda = (0.3,0.3,0.4)
   
    # print('lamda1,lamda2,lamda3:',lamda[0],lamda[1],lamda[2])
    # perplexity_li = linear_interpolation(dev_text, unigrams_prob_dict,bigram_prob_dict,
    # trigram_prob_dict,lamda[0],lamda[1],lamda[2])
    # print('liner interpolation perplexity:')
       
    # print(perplexity_li)
    # print('\n')

if __name__ == "__main__":
    main()

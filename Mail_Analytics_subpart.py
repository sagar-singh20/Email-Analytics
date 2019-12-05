#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:37:59 2019

@author: sinjinibose
"""
import email
import pandas as pd
from datetime import datetime
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#import spacy
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt


data1 = pd.read_csv('//Users//sinjinibose//Downloads//emails_with_tokens.csv')
data1 = data1.drop(data1.columns[0],axis=1)


import string
word_tokens=data1['tokens']


#word_tokens = word_tokens.apply(lambda x: x.lower())
word_tokens = word_tokens.apply(lambda x: x.replace("'",'').replace("[",'').replace("]",'').replace('-','').strip(' ').split(','))

#encoding digits as 'number' and time as 'time', links and email ids as 'email/link'
##There are some lines where there is no space assigned after a period and hence 
##two words have been treated as one token, therefore writing a check for all such 
##occurences and introducing a space between the two words
    
for num in range(0,len(word_tokens)):
    for n,i in enumerate(word_tokens[num]):
        if i.strip().isdigit() == True:
            word_tokens[num][n] = 'number'
        elif (i.strip().__contains__('am') or i.strip().__contains__('pm')):
            if(word_tokens[num][n-1].__contains__(':')):
                word_tokens[num][n-1] = 'time'
            word_tokens[num][n] = 'time'
        elif (i.strip().__contains__('.com') or i.strip().__contains__('@')):
            word_tokens[num][n] = 'email/link'
        elif (i.__contains__('.')):
            x = i.split('.')[1]
            y = i.split('.')[0]
            word_tokens[num][n] = y
            word_tokens[num].append(x)
        elif (len(i) == 1):
            word_tokens[num][n] = ''
            
        
            
##stripping leading and lagging spaces from words and appending them to a new list            
word_tokens_ = []
for i in word_tokens:
    temp = [j.strip(' ') for j in i]
    temp_1 = [i for i in temp if len(i)>2]
    word_tokens_.append(temp_1)
    
##creating pos_tags for the corpus
pos_tags = [nltk.pos_tag(i) for i in word_tokens_]

##removing NNPs, Prepositions, modular verbs or fillers, foreign words
NNP = []
CD = []
IN = []
MD = []

for i in pos_tags:
    for k,v in i:
        if v=='NNP':
            NNP.append(k)
        elif v=='CD':
            CD.append(k)
            
        if v=='MD':
            MD.append(k)
        if v=='IN':
            IN.append(k)
pos_tags_1 = pd.DataFrame()
pos_tags_1['proper_nouns'] = NNP
pos_tags_1.to_csv("//Users//sinjinibose//Documents//pos_tags_NNP.csv")

pos_tags_numbers = pd.DataFrame()
pos_tags_numbers['numbers'] = CD
pos_tags_numbers.to_csv("//Users//sinjinibose//Documents//pos_tags_CD.csv")

pos_tags_prepositions = pd.DataFrame()
pos_tags_prepositions['prepositions'] =  IN
pos_tags_prepositions.to_csv("//Users//sinjinibose//Documents//pos_tags_IN.csv")

pos_tags_verbs = pd.DataFrame()
pos_tags_verbs['Modal Verbs'] = MD
pos_tags_verbs.to_csv("//Users//sinjinibose//Documents//pos_tags_MD.csv")

word_tokens_2 =[]
count = 0
for i in word_tokens_:
    temp = [a for a in i if a not in NNP]
    temp_1 = [b for b in temp if b not in CD]
    temp_2 = [c for c in temp_1 if c not in IN]
    temp_3 = [d for d in temp_2 if d not in MD]
    temp_4 = [e.lower() for e in temp_3]
    word_tokens_2.append(temp_4)
    count = count + 1
    print(count)
    
#print(word_tokens_2)
##defining stopwords, and also customising the list
stopwords = []
from nltk.corpus import stopwords
stopwords = list(stopwords.words('english'))
punctuations = [i for i in string.punctuation]
greetings = ['Hey','Hello','Good','Morning','Greetings','Thanks','Thank','Regards','Warm','Best','Subject','From','Original','Message','""s""','""nt""','date','http','sent','time','would','could','call','number','click','david','jason','please','mailto','zufferli','""ll""','""ve""','fyi','john','paul','michael','todd','beth','andrew','joel','any','has']
for i in greetings:
    stopwords.append(i)
for i in punctuations:
    stopwords.append(i)
stopwords = [i.lower() for i in stopwords]

weekdays = ['sunday','monday','wednesday','thursday','friday']
months = ['january','february','march','april','may','june','july','august','september','october','november','december']


refined_tokens= []
for i in word_tokens_2:
    temp = [x for x in i if len(x) > 3]
    temp_1 = [x for x in temp if x not in weekdays]
    temp_2 = [x for x in temp_1 if x not in months]
    #print(temp_2)
    refined_tokens.append([j for j in temp_2 if j not in stopwords])



############Lemmatization#######################
from nltk.stem import WordNetLemmatizer 
Lemmatizer = WordNetLemmatizer()

###Using Wordnet since can't use spacy, providing pos_tags, however the tags used
## in wordnet are in the form of 'n' , 'v' etc. while the tags we get from pos 
## are in the format 'NN', 'VB' etc. so converting tags to the wordnet format with
##[j])[0][1][0].lower() , where j is a tag for a word. Also handling exception
## if any tag isn't found, the word is lemmatized without the tag, or assuming that
## it is a noun, which is the default tag for Wordnet.

try:
    for i in refined_tokens:
        for j in i:
            lemma = Lemmatizer.lemmatize(j,pos=nltk.pos_tag([j])[0][1][0].lower())
except KeyError:
    for i in refined_tokens:
        for j in i:
            lemma = Lemmatizer.lemmatize(j)
    


    

    
#import gensim
#import gensim.corpora as corpora
#from gensim.utils import simple_preprocess
#from gensim.models import CoherenceModel
##import spacy
#import pyLDAvis
#import pyLDAvis.gensim  
#import matplotlib.pyplot as plt
#
#from nltk.tag import StanfordPOSTagger
#stanford_dir = "//Users//sinjinibose//Downloads//stanford-postagger-full-2018-10-16//models//"
#modelfile = stanford_dir+"english-bidirectional-distsim.tagger"
#jarfile="//Users//sinjinibose//Downloads//stanford-postagger-full-2018-10-16//stanford-postagger-3.9.2.jar"
#tagger=StanfordPOSTagger(model_filename=modelfile, path_to_jar=jarfile)
#


####creating the corpora and dictionary for building the topic models'
id2word = corpora.Dictionary(refined_tokens)
texts = refined_tokens
corpus = [id2word.doc2bow(text) for text in texts]


#[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:10]]

from gensim.models.coherencemodel import CoherenceModel

n_topics = [10,15,20]

topics_coherence = []
for i in n_topics:
# Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=i,
                                           update_every=1,
                                           chunksize=50,
                                           passes=5,
                                           alpha='auto',
                                           per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    topics_coherence.append((lda_model,coherence_score))
perplexity_scores = [topics_coherence[i][0].log_perplexity(corpus) for i in range(0,3)]    
print(lda_model.print_topics())
doc_lda = lda_model[corpus]



#x = <gensim.models.ldamodel.LdaModel at 0x1b8f398748>
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(topics_coherence[0][0], corpus, id2word)
vis



print(topics_coherence)


# Print the Keyword in the 10 topics


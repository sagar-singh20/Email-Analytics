#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:37:59 2019

@author: sinjinibose
"""

import pandas as pd
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  

data1 = pd.read_csv('..//emails_with_tokens.csv')
data1 = data1.drop(data1.columns[0],axis=1)


import string


def get_clean_tokens(tokens):
    tokens = tokens.apply(lambda x: x.replace("'",'').replace("[",'').replace("]",'').replace('-','').strip(' ').split(','))
    
    #encoding digits as 'number' and time as 'time', links and email ids as 'email/link'
    ##There are some lines where there is no space assigned after a period and hence 
    ##two words have been treated as one token, therefore writing a check for all such 
    ##occurences and introducing a space between the two words
        
    for num in range(0,len(tokens)):
        for n,i in enumerate(tokens[num]):
            if i.strip().isdigit() == True:
                tokens[num][n] = 'number'
            elif (i.strip().__contains__('am') or i.strip().__contains__('pm')):
                if(tokens[num][n-1].__contains__(':')):
                    tokens[num][n-1] = 'time'
                tokens[num][n] = 'time'
            elif (i.strip().__contains__('.com') or i.strip().__contains__('@')):
                tokens[num][n] = 'email/link'
            elif (i.__contains__('.')):
                x = i.split('.')[1]
                y = i.split('.')[0]
                tokens[num][n] = y
                tokens[num].append(x)
            elif (len(i) == 1):
                tokens[num][n] = ''
    return tokens

word_tokens = data1['tokens']
word_tokens = get_clean_tokens(word_tokens)
                
            
            
##stripping leading and lagging spaces from words and appending them to a new list      

def strip_whitespaces(tok):      
    tokens_ = []
    for i in tok:
        temp = [j.strip(' ') for j in i]
        temp_1 = [i for i in temp if len(i)>2]
        word_tokens_.append(temp_1)
    return tokens_

word_tokens_= [strip_whitespaces(word_tokens)]
    
##pos tagging and parsing the tokens

pos_tags = [nltk.pos_tag(i) for i in word_tokens_]

##removing NNPs, Prepositions, modular verbs or fillers, foreign words


def parse_pos_tags(tags):
    prop_noun = []
    card_num = []
    modular_verb = []
    preposition = []
    foreign_word = []
    for i in tags:
        for k,v in i:
            if v=='NNP':
                prop_noun.append(k)
            elif v=='CD':
                card_num.append(k)
            if v=='MD':
                modular_verb.append(k)
            if v=='IN':
                preposition.append(k)
            if v=='FW':
                foreign_word.append(k)
    fin_list = [prop_noun,card_num,modular_verb,preposition,foreign_word]
    return fin_list

NNP = parse_pos_tags[0]
CD = parse_pos_tags[1]
IN = parse_pos_tags[2]
MD = parse_pos_tags[3]
FW = parse_pos_tags[4]


def remove_pos_tags (clean_tokens,NNP,CD,IN,MD):
    tokens=[]
    for i in clean_tokens[0:20000]:
        temp = [a for a in i if a not in NNP]
        temp_1 = [b for b in temp if b not in CD]
        temp_2 = [c for c in temp_1 if c not in IN]
        temp_3 = [d for d in temp_2 if d not in MD]
        temp_4 = [e.lower() for e in temp_3]
        tokens.append(temp_4)
    return tokens

word_tokens_2 = remove_pos_tags(word_tokens_,NNP,CD,IN,MD)


##defining stopwords, and also customising the list
stopwords = []
from nltk.corpus import stopwords
stopwords = list(stopwords.words('english'))
punctuations = [i for i in string.punctuation]

##adding custom stopwords to the list of stopwords
greetings = ['Hey','Hello','Good','Morning','Greetings','Thanks','Thank','Regards','Warm','Best','Subject','From','Original','Message','""s""','""nt""','date','http','sent','time','would','could','call','number','click','david','jason','please','mailto','zufferli','""ll""','""ve""','fyi','john','paul','michael','todd','beth','andrew','joel','any','has','"lar"']
for i in greetings:
    stopwords.append(i)
for i in punctuations:
    stopwords.append(i)
stopwords = [i.lower() for i in stopwords]

###defining a list for weekdays and months and including an extra check to remove such words
##If they haven't already been removed by the pos_tags (because of case mismatch)
weekdays = ['sunday','monday','wednesday','thursday','friday','tuesday']
months = ['january','february','march','april','may','june','july','august','september','october','november','december']

####Lemmatization and final cleaning for single tokens#####
from nltk.stem import WordNetLemmatizer 
Lemmatizer = WordNetLemmatizer()

def Lemmatization (parsed_tokens,weekdays,months,stopwords):
    tokens = []
    for i in word_tokens_2:
        temp = [x for x in i if len(x) > 3] 
        temp_1 = [x for x in temp if x not in weekdays]
        temp_2 = [x for x in temp_1 if x not in months]
        temp_3 = [j for j in temp_2 if j not in stopwords]
        try:
            lemma = [Lemmatizer.lemmatize(i,pos=nltk.pos_tag([i])[0][1][0].lower()) for i in temp_3]
        except :
            lemma = [Lemmatizer.lemmatize(i) for i in temp_3]
    
        tokens.append([j for j in lemma])

refined_tokens= Lemmatization(word_tokens_2,weekdays,months,stopwords)
        
    
##### bigram formation #####       
bigram_tokens = []
for i in range(0,len(refined_tokens)):
    bigram = []
    for j in range(0,len(refined_tokens[i])-1,2):
        if(len(refined_tokens[i]) > 0):
            bigram.append(refined_tokens[i][j]+" "+refined_tokens[i][j+1])
    bigram_tokens.append(bigram)



####creating the corpora and dictionary for building the topic models(single_tokens)
id2word = corpora.Dictionary(refined_tokens)
texts = refined_tokens
corpus = [id2word.doc2bow(text) for text in texts]


##no.of topics for which model is to be built
n_topics = [10,15,20]

# Building LDA model for 10,15,20 topics (single tokens)
topics_coherence = []
for i in n_topics:

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

###visualising with pyLDAvis for 10 topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(topics_coherence[0][0], corpus, id2word)
vis

#### dictionary and corpus for bigrams ####
id2word1 = corpora.Dictionary(bigram_tokens)
texts1 = bigram_tokens
corpus1 = [id2word1.doc2bow(text) for text in texts1]

topics_coherence_bigrams= []
for i in n_topics:
# Build LDA model for 10,15,20 topics for bigrams
    lda_model1 = gensim.models.ldamodel.LdaModel(corpus=corpus1,
                                           id2word=id2word1,
                                           num_topics=i,
                                           update_every=1,
                                           chunksize=50,
                                           passes=5,
                                           alpha='auto',
                                           per_word_topics=True)
    coherence_model_lda1 = CoherenceModel(model=lda_model1, texts=texts1, dictionary=id2word1, coherence='c_v')
    coherence_score1 = coherence_model_lda1.get_coherence()
    topics_coherence_bigrams.append((lda_model1,coherence_score1))

perplexity_scores_bigrams = [topics_coherence_bigrams[i][0].log_perplexity(corpus1) for i in range(0,3)]  

###visualising with pyLDAvis for 10 topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(topics_coherence_bigrams[0][0], corpus1, id2word1)
vis
###getting dominant topics in each document also percentage contribution
sent_topics_df = pd.DataFrame()
ldamodel = topics_coherence[0][0]
for i, row_list in enumerate(ldamodel[corpus]):
    row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
    for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:  # => dominant topic
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
        else:
            break
sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

contents = pd.Series(data1['message_body'][0:20000])
sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)





sent_topics_df1 = pd.DataFrame()
ldamodel1 = topics_coherence_bigrams[0][0]
for i, row_list in enumerate(ldamodel1[corpus1]):
    row = row_list[0] if ldamodel1.per_word_topics else row_list            
        
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
    for j, (topic_num, prop_topic) in enumerate(row):
        if j == 0:  # => dominant topic
            wp = ldamodel1.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            sent_topics_df1 = sent_topics_df1.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
        else:
            break
sent_topics_df1.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

contents = pd.Series(data1['message_body'][0:20000])
sent_topics_df1 = pd.concat([sent_topics_df1, contents], axis=1)
 


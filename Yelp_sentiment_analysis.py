#!/usr/bin/env python
# coding: utf-8

# In[12]:


# -------------------Part 1 of Yelp Project: Create a Class to Pre-Process reviews and ratings------------------

# First, import essential packages
import requests
import bs4
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import pandas as pd
sns.set()


# In[13]:


# Instantiate the stopwords and lemmatizer and modifiers

# We will extract stop words from a website, and then lemmatize them
wnl = WordNetLemmatizer()
allowed_modifiers = ['J', 'R', 'C']
nouns = ['N']
stopwords = []
stopwords_loc = "/Users/rabeya/Desktop/list_stopwords.csv"

# The code below will extract stopwords from a CSV file
import csv
with open(stopwords_loc, 'r', encoding='ISO-8859-1') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        stopwords.append(row)
        
stopwords = [w_[0] for w_ in stopwords ]


# In[14]:


# Can we lemmatize stopwords and create a unique list of the lemmatized stopwords?
len(stopwords)


# In[15]:


# This will covert Treebank Tags to Wordnet POS Tags

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('P'):
        return wordnet.NOUN
    else:
        return ''
    
lemmatized_stopwords = []
for pair in pos_tag(stopwords):
    wn_tag = get_wordnet_pos(pair[1])
    if wn_tag is not '':
        lemmed_word = wnl.lemmatize(pair[0], wn_tag)
        lemmatized_stopwords.append(lemmed_word)
    
    

    


# In[16]:


len(lemmatized_stopwords)


# In[17]:


final_stopwords = list(set(lemmatized_stopwords))
len(final_stopwords)


# In[18]:


#  -------- A function to remove contractions -------------
import re

def decontracted(phrase):
     
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[19]:


tokens = word_tokenize('You are not better than me')
tagged_ex = pos_tag(tokens)
words = []
for pair in tagged_ex:
    wn_tag = get_wordnet_pos(pair[1])
    if wn_tag is not '':
        lem_w = wnl.lemmatize(pair[0], wn_tag)
        words.append(lem_w)
    
    
lem_sent = ' '.join(words)
tokens


# In[20]:


# ----  Small function to create a unique version of a list
def uniquefy(L):
    W = []
    for x in L:
        if x not in W:
            W.append(x)
    return W

# ---------------- Create a class blueprint (and object) to track the pages and their reviews, ratings ------------
class YelpProcess:
    
    def __init__(self, link):
        self.link = link
        self.reviews = []
        self.ratings = []
        self.labels = []
   
        
    def process_reviews_ratings(self):
        page = requests.get(self.link)
        soup = BeautifulSoup(page.text, 'html.parser')
        
        # Extract the div-tags that contain the (MAIN) review content
        div_tags = soup.findAll('div', {'class': 'review-content' })
        
        # Extract all the paragraph <p lang="en"> tags (each one contains a review) from div-tags
        reviews_tag = [tag.p for tag in div_tags]
        reviews_tag_text = [t.get_text() for t in reviews_tag]
        
        # Extract all the corresponding ratings for each review
        ratings = [float(tag.div.div.div.img['alt'].split()[0]) for tag in div_tags]
        self.ratings = ratings
        
        # Now, the review has been loaded into Python from Yelp! Final Processing of reviews
        reviews_lower = [r.lower() for r in reviews_tag_text]
        reviews_no_contrs = [decontracted(r) for r in reviews_lower]
        reviews_tok = [word_tokenize(r) for r in reviews_no_contrs]
        reviews_words = [[w for w in token if w.isalpha() == True] for token in reviews_tok]
        reviews_pos = [pos_tag(W) for W in reviews_words]
        # code below removes nouns
        reviews_keywords = [[pair[0] for pair in P if pair[1][0] not in nouns] for P in reviews_pos]
        # code below removes stopwords
        reviews_edited = [[w for w in L if w not in final_stopwords] for L in reviews_keywords]
        # code below will lemmatize each review
        reviews_lemmed = []
        for rev_list in reviews_edited:
            lem_words = []
            tagged_rev = pos_tag(rev_list)
            for pair in tagged_rev:
                word = pair[0]
                wn_tag = get_wordnet_pos(pair[1])
                if wn_tag is not '':
                    lem_words.append(wnl.lemmatize(word, wn_tag))
            reviews_lemmed.append(lem_words)
        # remove stopwords one last time
        reviews_final_edit = [[w for w in M if w not in final_stopwords] for M in reviews_lemmed]
        # collect only unique words in each review-list
        reviews_unique = [uniquefy(T) for T in reviews_final_edit]
        # finally, concatenate and collect the processed reviews
        reviews = [' '.join(word_list) for word_list in reviews_unique]
        self.reviews = reviews
        
    
    
    def make_labels(self):
        labels = []
        for rate in self.ratings:
            if rate>= 4.0 and rate <= 5.0:
                labels.append(2)
            elif rate == 3.0:
                labels.append(1)
            elif rate < 3.0:
                labels.append(0)
        self.labels = labels
    
# We need the labels to be converted into numbers
# let 'Good' = 2
# let 'Average' = 1
# let 'Bad' = 0


# In[21]:


# This section shows the reviews and ratings data collected from Yelp

# Below are links to 2,900 Yelp restaurant reviews:
# https://www.yelp.com/biz/aki-japanese-restaurant-berkeley (Aki Japanese, 81 reviews)
# https://www.yelp.com/biz/kaze-ramen-berkeley-3 (Kaze Ramen, 375 reviews)
# https://www.yelp.com/biz/muraccis-berkeley-berkeley (Muracci's, 257 reviews)
# https://www.yelp.com/biz/tako-sushi-berkeley (Tako Sushi, 541 revs)
# https://www.yelp.com/biz/berkeley-thai-house-berkeley (Berkeley Thai House, 453 reviews)
# https://www.yelp.com/biz/mandarin-house-berkeley (Mandarin House, 198 reviews)
# https://www.yelp.com/biz/panchos-mexican-grill-berkeley (Panchos Mexican Grill, 154 reviews)
# https://www.yelp.com/biz/la-burrita-berkeley-2 (La Burrita, 235 reviews)
# https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4 (Kips Bar, 333 reviews)
# https://www.yelp.com/biz/buffet-fortuna-oakland (Buffett Fortuna, 275 reviews)
# https://www.yelp.com/biz/majikku-ramen-daly-city (Majikku Ramen, 563 reviews)

p1 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley"
p2 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley?start=20"
p3 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley?start=40"
p4 = "https://www.yelp.com/biz/aki-japanese-restaurant-berkeley?start=60"
p5 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley"
p6 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=20"
p7 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=40"
p8 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=60"
p9 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=80"
p10 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=100"
p11 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=120"
p12 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=140"
p13 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=160"
p14 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=180"
p15 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=200"
p16 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=220"
p17 = "https://www.yelp.com/biz/muraccis-berkeley-berkeley?start=240"
p18 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3"
p19 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=20"
p20 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=40"
p21 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=60"
p22 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=80"
p23 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=100"
p24 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=120"
p25 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=140"
p26 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=160"
p27 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=180"
p28 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=200"
p29 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=220"
p30 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=240"
p31 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=260"
p32 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=280"
p33 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=300"
p34 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=320"
p35 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=340"
p36 = "https://www.yelp.com/biz/kaze-ramen-berkeley-3?start=360"
p37 = "https://www.yelp.com/biz/tako-sushi-berkeley"
p38 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=20"
p39 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=40"
p40 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=60"
p41 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=80"
p42 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=100"
p43 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=120"
p44 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=140"
p45 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=160"
p46 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=180"
p47 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=200"
p48 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=220"
p49 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=240"
p50 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=260"
p51 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=280"
p52 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=300"
p53 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=320"
p54 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=340"
p55 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=360"
p56 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=380"
p57 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=400"
p58 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=420"
p59 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=440"
p60 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=460"
p61 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=480"
p62 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=500"
p63 = "https://www.yelp.com/biz/tako-sushi-berkeley?start=520"
p64 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley"
p65 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=20"
p66 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=40"
p67 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=60"
p68 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=80"
p69 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=100"
p70 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=120"
p71 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=140"
p72 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=160"
p73 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=180"
p74 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=200"
p75 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=220"
p76 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=240"
p77 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=260"
p78 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=280"
p79 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=300"
p80 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=320"
p81 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=340"
p82 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=360"
p83 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=380"
p84 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=400"
p85 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=420"
p86 = "https://www.yelp.com/biz/berkeley-thai-house-berkeley?start=440"
p87 = "https://www.yelp.com/biz/mandarin-house-berkeley"
p88 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=20"
p89 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=40"
p90 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=60"
p91 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=80"
p92 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=100"
p93 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=120"
p94 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=140"
p95 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=160"
p96 = "https://www.yelp.com/biz/mandarin-house-berkeley?start=180"
p97 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley"
p98 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=20"
p99 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=40"
p100 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=60"
p101 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=80"
p102 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=100"
p103 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=120"
p104 = "https://www.yelp.com/biz/panchos-mexican-grill-berkeley?start=140"
p105 = "https://www.yelp.com/biz/la-burrita-berkeley-2"
p106 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=20"
p107 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=40"
p108 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=60"
p109 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=80"
p110 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=100"
p111 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=120"
p112 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=140"
p113 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=160"
p114 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=180"
p115 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=200"
p116 = "https://www.yelp.com/biz/la-burrita-berkeley-2?start=220"
p117 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4"
p118 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=20"
p119 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=40"
p120 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=60"
p121 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=80"
p122 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=100"
p123 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=120"
p124 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=140"
p125 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=160"
p126 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=180"
p127 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=200"
p128 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=220"
p129 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=240"
p130 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=260"
p131 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=280"
p132 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=300"
p133 = "https://www.yelp.com/biz/kips-bar-and-grill-berkeley-4?start=320"
p134 = "https://www.yelp.com/biz/buffet-fortuna-oakland"
p135 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=20"
p136 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=40"
p137 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=60"
p138 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=80"
p139 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=100"
p140 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=120"
p141 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=140"
p142 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=160"
p143 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=180"
p144 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=200"
p145 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=220"
p146 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=240"
p147 = "https://www.yelp.com/biz/buffet-fortuna-oakland?start=260"
p148 = "https://www.yelp.com/biz/majikku-ramen-daly-city"
p148 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=20"
p149 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=40"
p150 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=60"
p151 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=80"
p152 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=100"
p153 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=120"
p154 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=140"
p155 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=160"
p156 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=180"
p157 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=200"
p158 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=220"
p159 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=240"
p160 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=260"
p161 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=280"
p162 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=300"
p163 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=320"
p164 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=340"
p165 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=360"
p166 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=380"
p167 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=400"
p168 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=420"
p169 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=440"
p170 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=460"
p171 = "https://www.yelp.com/biz/majikku-ramen-daly-city?start=480"




pages = [p1,p2,p3,p4,p5,p6,p7,p8,p9,
         p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,
         p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,
         p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,
         p40,p41,p42,p43,p44,p45,p46,p47,p48,p49,
         p50,p51,p52,p53,p54,p55,p56,p57,p58,p59,
         p60,p61,p62,p63,p64,p65,p66,p67,p68,p69,
         p70,p71,p72,p73,p74,p75,p76,p77,p78,p79,
         p80,p81,p82,p83,p84,p85,p86,p87,p88,p89,
         p90,p91,p92,p93,p94,p95,p96,p97,p98,p99,
         p100,p101,p102,p103,p104,p105,p106,p107,p108,p109,
         p110,p111,p112,p113,p114,p115,p116,p117,p118,p119,
         p120,p121,p122,p123,p124,p125,p126,p127,p128,p129,
         p130,p131,p132,p133,p134,p135,p136,p137,p138,p139,
         p140,p141,p142,p143,p144,p145,p146,p147,p148,p149, 
         p150,p151,p152,p153,p154,p155,p146,p147,p148,p149,
         p160,p161,p162,p163,p164,p165,p166,p167,p168,p169,
         p171 ]


# In[22]:


#----------------------- Part 2 of Yelp Project: Analysis of Reviews and matching ratings ------------------------

# ---------------- Section 1: Make the DataFrame --------------------

# Now that we have created the initial website reviews/ratings Class structure,
# we need to input the reviews and ratings into a Pandas Dataframe


def review_setup(link):
    R = YelpProcess(link)
    R.process_reviews_ratings()
    R.make_labels()
    return R
    
def review_objects(links):
    result = []
    for i in range(0, len(links)):
        result.append(review_setup(links[i]))
    return result
        
def review_frame(Obj):
    df = pd.DataFrame({'reviews': Obj.reviews, 'opinions': Obj.labels})
    return df


def make_frames(obj_list):
    frames=[]
    for obj in obj_list:
        frames.append(review_frame(obj))
    return frames

link_objs = review_objects(pages)
all_reviews = pd.concat(make_frames(link_objs)).reset_index(drop=True)                  


# In[ ]:


print(all_reviews['opinions'].value_counts())
orig = list(all_reviews['opinions'].value_counts())
print([s/sum(orig) for s in orig])


# In[ ]:


1281+1135+975


# In[ ]:


all_reviews


# In[ ]:


# -------------- This section is CountVectorizer: DTM Matrix (word-frequency) Analysis --------------
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(max_features=5000, binary=True)
sparse_dtm = countvec.fit_transform(all_reviews['reviews'])
features_dtm = sparse_dtm.toarray()
DTM_table = pd.DataFrame(features_dtm, columns=countvec.get_feature_names())


# In[ ]:


# ---------------- Now we will use the Logistic (Multi-Class) Classifier and create our training/test data -------

response = all_reviews['opinions'].values

# Part 1: First, we will do our Logit Classifier on the DTM features matrix (Count Vectorizer) ---
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

X_dtm, y_dtm = features_dtm, response

# First we need to split our data into training and testing sets
X_train_dtm, X_test_dtm, y_train_dtm, y_test_dtm = train_test_split(X_dtm, y_dtm, train_size=0.8, random_state=0)

# Normalize training data with the scaler
scaler = StandardScaler()
X_train_dtm_scaled = scaler.fit_transform(X_train_dtm)
X_test_dtm_scaled = scaler.fit_transform(X_test_dtm)


# In[ ]:


# OVR and OVO logistic classifiers for DTM data
logit_dtm_OVR = OneVsRestClassifier(LogisticRegressionCV())
logit_dtm_OVO = OneVsOneClassifier(LogisticRegressionCV())
OVR = logit_dtm_OVR.fit(X_train_dtm_scaled, y_train_dtm)
OVO = logit_dtm_OVO.fit(X_train_dtm_scaled, y_train_dtm)

OVR_score = logit_dtm_OVR.score(X_test_dtm_scaled, y_test_dtm)
OVO_score = logit_dtm_OVO.score(X_test_dtm_scaled, y_test_dtm)
print("Logistic (OVO) accuracy (DTM): {}%".format(round(100*OVR.score(X_test_dtm_scaled, y_test_dtm), 3)))
print("Logistic accuracy (OVR) (DTM): {}%".format(round(100*OVO.score(X_test_dtm_scaled, y_test_dtm), 3)))


# In[ ]:


from sklearn.metrics import recall_score, precision_score
import numpy as np

y_test_tfidf_PD = pd.DataFrame(y_test_tfidf)
base_rate_dtm = list(100*(y_test_tfidf_PD[0].value_counts().head(1) / len(y_test_tfidf_PD)))[0]

print("Base Rate Accuracy: {}%".format(round(base_rate_dtm, 3)))
print("OVR (DTM) Recall Scores: ", recall_score(y_test_dtm, OVR.predict(X_test_dtm), average=None))
print("OVR (DTM) Precision Scores: ", precision_score(y_test_dtm, OVR.predict(X_test_dtm), average=None))


# In[ ]:


# This part is for graphing the accuracy of the OVR-classifier (DTM)
labels_dtm_ovr = OVR.classes_
y_pred_dtm_ovr = OVR.predict(X_test_dtm_scaled)
cm_dtm_ovr = confusion_matrix(y_test_dtm, y_pred_dtm_ovr)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_ovr, annot=True, xticklabels=labels_dtm_ovr, yticklabels=labels_dtm_ovr)
plt.xlabel('Logistic Prediction (OVR, DTM features)')
plt.ylabel('Truth')


# In[ ]:


print("Base Rate Accuracy: {}%".format(round(base_rate_dtm, 3)))
print("OVO (DTM) Recall Scores: ", recall_score(y_test_dtm, OVO.predict(X_test_dtm), average=None))
print("OVO (DTM) Precision Scores: ", precision_score(y_test_dtm, OVO.predict(X_test_dtm), average=None))


# In[ ]:


# This part is for graphing the accuracy of the OVO-classifier (DTM)
labels_dtm_ovo = OVO.classes_
y_pred_dtm_ovo = OVO.predict(X_test_dtm)
cm_dtm_ovo = confusion_matrix(y_test_dtm, y_pred_dtm_ovo)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_ovo, annot=True, xticklabels=labels_dtm_ovo, yticklabels=labels_dtm_ovo)
plt.xlabel('Logistic Prediction (OVO, DTM features)')
plt.ylabel('Truth')


# In[ ]:


# ---------------- Section 2: TF-IDF Matrix Vectorizer to analyze Word importance -----------------

# First, create the TF-IDF Vectorizer Object
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
sparse_tfidf = tfidfvec.fit_transform(all_reviews['reviews'])

#Next, create the TF-IDF feature matrix
#tfidf = pd.DataFrame(sparse_tfidf.toarray(), columns=tfidfvec.get_feature_names(), index=all_reviews.index)
features_tfidf = sparse_tfidf.toarray()


# In[25]:


import pickle


# In[30]:


pickle_in = open("TFIDF_reviews.pickle", 'rb')
tfidf_matrix = pickle.load(pickle_in)
tfidf_matrix.head()


# In[33]:


pickle_in_revs = open("yelp_reviews.pickle", 'rb')
all_reviews = pickle.load(pickle_in_revs)
all_reviews['opinions'].head()


# In[35]:


new_tfidf = tfidf_matrix.copy()
new_tfidf['labels_'] = all_reviews['opinions']
new_tfidf.head()


# In[44]:


good = new_tfidf[new_tfidf['labels_'] == 2]
good.max(numeric_only=True).sort_values(ascending=False)


# In[45]:


all_reviews['reviews']


# In[ ]:





# In[ ]:





# In[ ]:


# Part 2: Now, we will do our Logit Classifier on the TF-IDF features matrix ---------
# We are hoping to use word features from the TF-IDF matrix to help
# First we need to split our data into training and testing sets
X_tfidf, y_tfidf = features_tfidf, response

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_tfidf, train_size=0.8, random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression
# Logit Classifier for TF-IDF features/data
logit_tfidf_OVR = OneVsRestClassifier(LogisticRegression(max_iter=2000))
logit_tfidf_OVO = OneVsOneClassifier(LogisticRegression(max_iter=2000))
OVR = logit_tfidf_OVR.fit(X_train_tfidf, y_train_tfidf)
OVO = logit_tfidf_OVO.fit(X_train_tfidf, y_train_tfidf)
# Scoring
OVR_score = logit_tfidf_OVR.score(X_test_tfidf, y_test_tfidf)
OVO_score = logit_tfidf_OVO.score(X_test_tfidf, y_test_tfidf)
# Print accuracies
print("Logistic accuracy (OVR, TF-IDF features): {}%".format(round(100*OVR.score(X_test_tfidf, y_test_tfidf), 3)))
print("Logistic accuracy (OVO, TF-IDF features): {}%".format(round(100*OVO.score(X_test_tfidf, y_test_tfidf), 3)))


# In[127]:


from sklearn.metrics import recall_score, precision_score
import numpy as np
y_test_tfidf_PD = pd.DataFrame(y_test_tfidf)
base_rate_tfidf = list(100*(y_test_tfidf_PD[0].value_counts().head(1) / len(y_test_tfidf_PD)))[0]

print("Base Rate Accuracy: {}%".format(round(base_rate_tfidf,3)))
print("Recall Scores: ", recall_score(y_test_tfidf, OVR.predict(X_test_tfidf), average=None))
print("Precision Scores: ", precision_score(y_test_tfidf, OVR.predict(X_test_tfidf), average=None))


# In[128]:


# This last part is for graphing the accuracy of the classifier (TF-IDF) using OVR classification
labels_tfidf_ovr = OVR.classes_
y_pred_tfidf_ovr = OVR.predict(X_test_tfidf)
cm_tfidf_ovr = confusion_matrix(y_test_tfidf, y_pred_tfidf_ovr)

plt.figure(figsize=(10,7))
sns.heatmap(cm_tfidf_ovr, annot=True, xticklabels=labels_tfidf_ovr, yticklabels=labels_tfidf_ovr)
plt.xlabel('Logistic Prediction (TF-IDF features, OVR)')
plt.ylabel('Truth')


# In[129]:


# This last part is for graphing the accuracy of the classifier (TF-IDF) using OVO classification

labels_tfidf_ovo = OVO.classes_
y_pred_tfidf_ovo = OVO.predict(X_test_tfidf)
cm_tfidf_ovo = confusion_matrix(y_test_tfidf, y_pred_tfidf_ovo)

plt.figure(figsize=(10,7))
sns.heatmap(cm_tfidf_ovo, annot=True, xticklabels=labels_tfidf_ovo, yticklabels=labels_tfidf_ovo)
plt.xlabel('Logistic Prediction (OVO, TF-IDF features)')
plt.ylabel('Truth')


# In[ ]:





# In[ ]:





# In[327]:


# -------------- Now we will try doing classification based on SVM Classifier ------------


# In[130]:


# SVM stands for Support Vector Machine algorithm

# -------------- This section is CountVectorizer: DTM Matrix (word-frequency) Analysis --------------
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(max_features=5000, binary=True)
sparse_dtm = countvec.fit_transform(all_reviews['reviews'])
features_dtm = sparse_dtm.toarray()
response = all_reviews['opinions'].values

X_dtm, y_dtm = features_dtm, response

# First we will do the SVM classification with the DTM-feature matrix
#First, split the X_dtm and y_dtm data

X_train_dtm_svm, X_test_dtm_svm, y_train_dtm_svm, y_test_dtm_svm = train_test_split(X_dtm, y_dtm, train_size=0.8, random_state=2)


# In[131]:


# Next, import the SVM classifier and set up the model
# We will need to do OVR and OVO separate models from the SVM structure
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

# OVR SVM classifier for DTM features matrix
clf_svm_dtm_ovr = OneVsRestClassifier(svm.SVC(gamma=0.003, C=200))
clf_svm_dtm_ovr.fit(X_train_dtm_svm, y_train_dtm_svm)
print("SVM (OVR, DTM-features) Accuracy: {}%".format(round(100*clf_svm_dtm_ovr.score(X_test_dtm_svm, y_test_dtm_svm), 3)))


# In[132]:


# OVO SVM classifier for DTM features matrix
clf_svm_dtm_ovo = OneVsOneClassifier(svm.SVC(gamma=0.003, C=200))
clf_svm_dtm_ovo.fit(X_train_dtm_svm, y_train_dtm_svm)
print("SVM (OVO, DTM-features) Accuracy: {}%".format(round(100*clf_svm_dtm_ovo.score(X_test_dtm_svm, y_test_dtm_svm), 3)))


# In[133]:


from sklearn.metrics import recall_score, precision_score
import numpy as np

y_test_dtm_svm_PD = pd.DataFrame(y_test_dtm_svm)
base_rate_svm_dtm = list(100*(y_test_dtm_svm_PD[0].value_counts().head(1) / len(y_test_dtm_svm_PD)))[0]

print("Base Rate Accuracy: {}%".format(round(base_rate_dtm, 3)))
print("OVR (DTM) SVM-Recall Scores: ", recall_score(y_test_dtm_svm, clf_svm_dtm_ovr.predict(X_test_dtm_svm), average=None))
print("OVR (DTM) SVM-Precision Scores: ", precision_score(y_test_dtm_svm, clf_svm_dtm_ovr.predict(X_test_dtm_svm), average=None))


# In[ ]:


y_test_dtm_svm_PD = pd.DataFrame(y_test_dtm_svm)
base_rate_svm_dtm = list(100*(y_test_dtm_svm_PD[0].value_counts().head(1) / len(y_test_dtm_svm_PD)))[0]

print("Base Rate Accuracy: {}%".format(round(base_rate_dtm, 3)))
print("OVO (DTM) SVM-Recall Scores: ", recall_score(y_test_dtm_svm, clf_svm_dtm_ovo.predict(X_test_dtm_svm), average=None))
print("OVO (DTM) SVM-Precision Scores: ", precision_score(y_test_dtm_svm, clf_svm_dtm_ovo.predict(X_test_dtm_svm), average=None))


# In[134]:


# This part is for graphing the accuracy of the OVR SVM-classifier (DTM features)
labels_dtm_svm_ovr = clf_svm_dtm_ovr.classes_
y_pred_dtm_svm_ovr = clf_svm_dtm_ovr.predict(X_test_dtm_svm)
cm_dtm_svm_ovr = confusion_matrix(y_test_dtm_svm, y_pred_dtm_svm_ovr)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_svm_ovr, annot=True, xticklabels=labels_dtm_svm_ovr, yticklabels=labels_dtm_svm_ovr)
plt.xlabel('SVM Prediction (OVR classifier, DTM features)')
plt.ylabel('Truth')


# In[135]:


# This part is for graphing the accuracy of the OVO SVM-classifier (DTM features)
labels_dtm_svm_ovo = clf_svm_dtm_ovo.classes_
y_pred_dtm_svm_ovo = clf_svm_dtm_ovo.predict(X_test_dtm_svm)
cm_dtm_svm_ovo = confusion_matrix(y_test_dtm_svm, y_pred_dtm_svm_ovo)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_svm_ovo, annot=True, xticklabels=labels_dtm_svm_ovo, yticklabels=labels_dtm_svm_ovo)
plt.xlabel('SVM Prediction (OVO classifier, DTM features)')
plt.ylabel('Truth')


# In[ ]:





# In[344]:


# Now we will do OVR and OVO SVM classifiers using the TF-IDF features matrix ---------


# In[136]:


# First, create the TF-IDF Vectorizer Object
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
sparse_tfidf = tfidfvec.fit_transform(all_reviews['reviews'])

#Next, create the TF-IDF feature matrix
features_tfidf = sparse_tfidf.toarray()

# Part 2: Now, we will do our SVM Classifier on the TF-IDF features matrix ---------
# First we need to split our data into training and testing sets
response = all_reviews['opinions'].values
X_tfidf, y_tfidf = features_tfidf, response

# Then, split the X_tfidf and y_tfidf data
X_train_tfidf_svm, X_test_tfidf_svm, y_train_tfidf_svm, y_test_tfidf_svm = train_test_split(X_tfidf, y_tfidf, train_size=0.8, random_state=3)


# In[137]:


# OVR SVM classifier for TF-IDF features matrix
clf_svm_tfidf_ovr = OneVsRestClassifier(svm.SVC(gamma=0.003, C=200))
clf_svm_tfidf_ovr.fit(X_train_tfidf_svm, y_train_tfidf_svm)
print("SVM (OVR, TF-IDF-features) Accuracy: {}%".format(round(100*clf_svm_tfidf_ovr.score(X_test_tfidf_svm, y_test_tfidf_svm), 3)))


# In[138]:


# OVO SVM classifier for TF-IDF features matrix
clf_svm_tfidf_ovo = OneVsOneClassifier(svm.SVC(gamma=0.003, C=200))
clf_svm_tfidf_ovo.fit(X_train_tfidf_svm, y_train_tfidf_svm)
print("SVM (OVO, TF-IDF features) Accuracy: {}%".format(round(100*clf_svm_tfidf_ovo.score(X_test_tfidf_svm, y_test_tfidf_svm), 3)))


# In[139]:


y_test_tfidf_svm_PD = pd.DataFrame(y_test_tfidf_svm)
base_rate_svm_tfidf = list(100*(y_test_tfidf_svm_PD[0].value_counts().head(1) / len(y_test_tfidf_svm_PD)))[0]

print("Base Rate Accuracy: {}%".format(round(base_rate_svm_tfidf, 3)))
print("OVO (TF-IDF) SVM-Recall Scores: ", recall_score(y_test_tfidf_svm, clf_svm_tfidf_ovo.predict(X_test_tfidf_svm), average=None))
print("OVO (TF-IDF) SVM-Precision Scores: ", precision_score(y_test_tfidf_svm, clf_svm_tfidf_ovo.predict(X_test_tfidf_svm), average=None))


# In[140]:


y_test_tfidf_svm_PD = pd.DataFrame(y_test_tfidf_svm)
base_rate_svm_tfidf = list(100*(y_test_tfidf_svm_PD[0].value_counts().head(1) / len(y_test_tfidf_svm_PD)))[0]

print("Base Rate Accuracy: {}%".format(round(base_rate_svm_tfidf, 3)))
print("OVR (TF-IDF) SVM-Recall Scores: ", recall_score(y_test_tfidf_svm, clf_svm_tfidf_ovr.predict(X_test_tfidf_svm), average=None))
print("OVR (TF-IDF) SVM-Precision Scores: ", precision_score(y_test_tfidf_svm, clf_svm_tfidf_ovr.predict(X_test_tfidf_svm), average=None))


# In[141]:


# This part is for graphing the accuracy of the SVM OVO classifier (TF-IDF features)
labels_tfidf_svm_ovo = clf_svm_tfidf_ovo.classes_
y_pred_tfidf_svm_ovo = clf_svm_tfidf_ovo.predict(X_test_tfidf_svm)
cm_svm_tfidf_ovo = confusion_matrix(y_test_tfidf_svm, y_pred_tfidf_svm_ovo)

plt.figure(figsize=(10,7))
sns.heatmap(cm_svm_tfidf_ovo, annot=True, xticklabels=labels_tfidf_svm_ovo, yticklabels=labels_tfidf_svm_ovo)
plt.xlabel('SVM Prediction (OVO classifier, TF-IDF features)')
plt.ylabel('Truth')


# In[142]:


# This part is for graphing the accuracy of the OVR SVM-classifier (TF-IDF features)
labels_tfidf_svm_ovr = clf_svm_tfidf_ovr.classes_
y_pred_tfidf_svm_ovr = clf_svm_tfidf_ovr.predict(X_test_tfidf_svm)
cm_tfidf_svm_ovr = confusion_matrix(y_test_tfidf_svm, y_pred_tfidf_svm_ovr)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_svm_ovr, annot=True, xticklabels=labels_tfidf_svm_ovr, yticklabels=labels_tfidf_svm_ovr)
plt.xlabel('SVM Prediction (OVR classifier,TF-IDF features)')
plt.ylabel('Truth')


# In[146]:


tfidf_df = pd.DataFrame(features_tfidf, columns=tfidfvec.get_feature_names(), index=all_reviews.index)


# In[147]:


tfidf_df.head()


# In[149]:


tfidf_df['opinions_'] = all_reviews['opinions']
tfidf_df.head()


# In[153]:


good = tfidf_df[tfidf_df['opinions_']==2]
average = tfidf_df[tfidf_df['opinions_']==1]
not_good = tfidf_df[tfidf_df['opinions_']==0]


# In[156]:


good.max(numeric_only=True).sort_values(ascending=False).head(15)


# In[157]:


not_good.max(numeric_only=True).sort_values(ascending=False).head(15)


# In[158]:


average.max(numeric_only=True).sort_values(ascending=False).head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[136]:


# ------- Let's do some Data Preprocessing on this with the FastAI library ------- #


# In[137]:


from fastai.imports import *
from fastai_structured import *

from pandas_summary import DataFrameSummary


# In[138]:


data_DTM_raw = pd.concat([DTM_table, all_reviews['opinions']], axis=1)


# In[139]:


data_DTM_raw.head()


# In[140]:


train_data_DTM_raw, test_data_DTM_raw = train_test_split(data_DTM_raw, train_size = 0.9, random_state=42)


# In[141]:


train_data_DTM_raw.shape, test_data_DTM_raw.shape


# In[142]:


train_data_DTM_raw.head(10)


# In[143]:


# converts category strings into Pandas categorical data type
# for the training set
train_cats(train_data_DTM_raw)


# In[ ]:





# In[144]:


# Pre-processing step

df, y, nas = proc_df(train_data_DTM_raw, 'opinions')


# In[145]:


df.head()


# In[146]:


y


# In[147]:


test_data_DTM_raw.shape


# In[148]:


df.shape


# In[164]:


def split_vals(a,n): 
    return a[:n].copy(), a[n:].copy()


n_valid = test_data_DTM_raw.shape[0]//2  # same as the # of rows in test data
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(train_data_DTM_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape


# In[165]:


# Base model of Random Forest (with training and validation sets)
from sklearn.ensemble import RandomForestClassifier


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
    
    
m1 = RandomForestClassifier(n_jobs=-1, n_estimators=20, oob_score=False)
get_ipython().run_line_magic('time', 'm1.fit(X_train, y_train)')
print_score(m1)


# In[166]:


m2 = RandomForestClassifier(n_jobs=-1, n_estimators=30, oob_score=False)
get_ipython().run_line_magic('time', 'm2.fit(X_train, y_train)')
print_score(m2)


# In[167]:


m3 = RandomForestClassifier(n_jobs=-1, n_estimators=40, min_samples_leaf=3, oob_score=False)
get_ipython().run_line_magic('time', 'm3.fit(X_train, y_train)')
print_score(m3)


# In[175]:


preds_m3 = np.stack([t.predict(X_valid) for t in m3.estimators_])
preds_m3[:,0], max([2, 1,0], key = lambda i: list(preds[:,0]).count(i)), y_valid[0]


# In[176]:


preds_m2 = np.stack([t.predict(X_valid) for t in m2.estimators_])


# In[177]:


preds_m1 = np.stack([t.predict(X_valid) for t in m1.estimators_])


# In[179]:


from sklearn import metrics
print(preds_m1.shape)
print(preds_m2.shape)
print(preds_m3.shape)


# In[181]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds_m1[:i+1], axis=0)) for i in range(len(m1.estimators_))]);


# In[182]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds_m2[:i+1], axis=0)) for i in range(len(m2.estimators_))]);


# In[183]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds_m3[:i+1], axis=0)) for i in range(len(m3.estimators_))]);


# In[ ]:





# In[ ]:





# In[ ]:





# In[185]:


# Display the corresponding confusion matrix (validation set)

labels_dtm_rf = m3.classes_
y_valid_pred = m3.predict(X_valid)
cm_dtm_rf = confusion_matrix(y_valid, y_valid_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm_dtm_rf, annot=True, xticklabels=labels_dtm_rf, yticklabels=labels_dtm_rf)
plt.xlabel('R.F. (DTM features) Validation Set')
plt.ylabel('Truth')


# In[ ]:





# In[ ]:





# In[ ]:





# In[159]:


# Question: can we engineer a NEW dataset with new features,
# but derived from the old data?

# This new dataset would have 4 features:
# - 1) #(pos adj) / #(neg adj) 
# - 2) #(pos adv) / #(neg adv) 
# - 3) #(pos verb) / #(neg verb)
# - 4) # miscellanous words (figure this out in detail)

# Feature 1: first we need a list of almost every (+) english adjective, lemmatized

# First, we still need to process all the reviews and remove stopwords, unnecsary words 
# (which ontribute no meaning), lemmatize each word according to its pos_tag, and actually
# pos_tag the words

# Then when we process the reviews, we need to use Word2Vec to compare them to
# bag of positive and negative words in our vocabulary


# In[215]:


# imports needed and logging
import gzip
import multiprocessing

from gensim.models import Word2Vec 
import logging

reviews_list = (list(all_reviews['reviews']))


# In[246]:


# documents = list(all_reviews['reviews']) + positive words + negative words
positive_words = []
negative_words = []

with open("/Users/rabeya/Desktop/positive_words_NLP.csv", 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        positive_words.append(row)
        
positive_words = [w[0] for w in positive_words]

with open("/Users/rabeya/Desktop/negative_words_NLP.csv", 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        negative_words.append(row)
        
negative_words = [w[0] for w in negative_words]

print(len(positive_words))
print(len(negative_words))
#positive_words


# In[261]:


positive_words_container = [[w] for w in positive_words]
negative_words_container = [[w] for w in negative_words]
reviews_list_container = [[L] for L in reviews_list]


# In[375]:


positive_words


# In[404]:


def lemma_words(words):
    lemmatized_stopwords = []
    for pair in pos_tag(words):
        wn_tag = get_wordnet_pos(pair[1])
        word = pair[0]
        if wn_tag is not '':
            lemmed_word = wnl.lemmatize(word, wn_tag)
            lemmatized_stopwords.append(word)
        
    return lemmatized_stopwords

lemmed_pos_words = list(set(lemma_words(positive_words)))
lemmed_neg_words = list(set(lemma_words(negative_words)))

print("All (+) words: ", len(positive_words))
print("Lemmed (+) words: ", len(lemmed_pos_words))

print("All (-) words: ", len(negative_words))
print("Lemmed (-) words: ", len(lemmed_neg_words))


# In[406]:


positive_words


# In[403]:


lemmed_neg_words


# In[23]:


# let's import pickle and save the all_reviews dataframe

import pickle
pickle_out_reviews = open('yelp_reviews.pickle', 'wb')
pickle.dump(all_reviews, pickle_out_reviews)
pickle_out_reviews.close()


# In[20]:


# -------------- This section is CountVectorizer: DTM Matrix (word-frequency) Analysis --------------
from sklearn.feature_extraction.text import CountVectorizer

countvec = CountVectorizer(max_features=5000, binary=True)
sparse_dtm = countvec.fit_transform(all_reviews['reviews'])
features_dtm = sparse_dtm.toarray()
response = all_reviews['opinions'].values

X_dtm, y_dtm = features_dtm, response
DTM_table = pd.DataFrame(features_dtm, columns=countvec.get_feature_names())


# In[24]:


pickle_out_DTM = open('DTMatrix_reviews.pickle', 'wb')
pickle.dump(DTM_table, pickle_out_DTM)
pickle_out_DTM.close()


# In[28]:


# First, create the TF-IDF Vectorizer Object
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfvec = TfidfVectorizer()
sparse_tfidf = tfidfvec.fit_transform(all_reviews['reviews'])

#Next, create the TF-IDF feature matrix
tfidf_dframe = pd.DataFrame(sparse_tfidf.toarray(), columns=tfidfvec.get_feature_names(), index=all_reviews.index)
tfidf_dframe.head()


# In[30]:


pickle_out_TFIDF = open('TFIDF_reviews.pickle', 'wb')
pickle.dump(tfidf_dframe, pickle_out_TFIDF)
pickle_out_TFIDF.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[177]:


# K Nearest-Neighbors Classifier
import numpy as np
# Let's do a toy example on the Wisconsin 699-sample breast cancer dataset from UC Irvine

cancer_data = pd.read_csv('breast-cancer-wisconsin_data.txt')


# In[35]:


cancer_data.head()


# In[37]:


headers = {"1000025": "ID", "5": "Clump Rad.", '1':'Cell Size', '1.1':'Cell Shape', 
          '1.2':'Adhesion', '2':'Epithelial Size', '1.3':'Nuclei Size', '3':'Chromatin', 
          '1.4':'Nucleoli Size', '1.5':'Mitoses', '2.1':'Type'}
cancer_data.rename(index=str, columns=headers, inplace=True)


# In[39]:


# in this dataset, 2 = Benign Tumor, 4 = Malignant Tumor
cancer_data.head()


# In[40]:


cancer_data_noID = cancer_data.drop('ID', axis=1)


# In[46]:


cancer_data_noID.head()


# In[51]:


X = cancer_data_noID.drop('Type', axis=1, inplace=False)
y = cancer_data_noID['Type']


# In[53]:


X.head()


# In[108]:


# Before we even do training and modeling, we need to replace '?' and other missing values

cancer_data_noID.head(25)


# In[107]:


cancer_data_noID.isin(['?'])


# In[109]:


cancer_data_noID.columns


# In[121]:


all(cancer_data_noID['Type'].isin(['?']) == False)


# In[123]:


cancer_data_noID['Nuclei Size'][0]


# In[131]:


# the str.isdigit() method checks if the string contains only digits
'?'.isdigit()


# In[152]:


def check_non_numbers(iterable):
    non_numbers = []
    for t in iterable:
        if type(t)==int:
            continue
        if type(t)==str:
            if not t.isdigit():
                non_numbers.append(t)   
    return non_numbers

list_of_missing_in_feature = {}

for feature in list(cancer_data_noID.columns):
    list_of_missing_in_feature[feature] = check_non_numbers(cancer_data_noID[feature])
    


# In[153]:


list_of_missing_in_feature


# In[155]:


len(cancer_data_noID['Nuclei Size'])


# In[158]:


nuclei_series = cancer_data_noID['Nuclei Size'].isin(['?']) 


# In[178]:


non_numbers_nuclei = nuclei_series[nuclei_series==False]
digit_indices = np.array([int(b) for b in non_numbers_nuclei.index])


# In[183]:


# This removes all the rows with "?" (missing vaules) in them
# Luckily, they represent only 2% of the data
cancer_data_noID_removed = cancer_data_noID.iloc[digit_indices, :]


# In[191]:


cancer_data_noID_removed.head()


# In[ ]:





# In[215]:


# Let's first split the data into "real" and "model" data sets
# Then we will do K-Fold cross-validation on the "model" set

from sklearn.model_selection import train_test_split
model_set, real_set = train_test_split(cancer_data_noID_removed, test_size=0.15, random_state=40)


# In[216]:


len(model_set)


# In[217]:


# Let's first do cross-validation for Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold

# prepare cross validation
# will create 4 groups, will shuffle data (True-value), and random seed of 1
#kfold = KFold(4, True, 1)
#RepeatedKFold??

rkf = RepeatedKFold(n_splits=4, n_repeats=10, random_state=1)

# Now we obtain X and y (for doing cross-validation) from the "MODEL" set (85% of the whole dataset)
X, y = model_set.drop('Type', axis=1, inplace=False), model_set['Type']


# In[218]:


len(X)


# In[219]:


rkf.split(X.values)


# In[220]:


for train_index, test_index in rkf.split(X.values):
    print("TRAIN:", train_index, "TEST:", test_index)


# In[240]:


def logitscore(X_train, y_train, test_X, actual):
    LR = LogisticRegression(C=5, max_iter=500, solver='lbfgs')
    LR.fit(X_train, y_train)
    score = LR.score(test_X, actual)
    return score

cv_scores = []
for train_index, test_index in rkf.split(X.values):
    X_train = X.iloc[train_index, :].values
    y_train = y[train_index].values
    X_test = X.iloc[test_index, :].values
    y_truth = y[test_index].values
    fold_score = logitscore(X_train, y_train, X_test, y_truth)
    cv_scores.append(fold_score)
    
    
    


# In[241]:


cv_scores


# In[242]:


np.mean(cv_scores)


# In[243]:


np.std(cv_scores)


# In[244]:


real_X = real_set.drop('Type', axis=1, inplace=False)
real_y = real_set['Type']


# In[249]:


LR_real = LogisticRegression(C=5, max_iter=500, solver='lbfgs')
LR_real.fit(X.values, y.values)
score_realdata = LR_real.score(real_X.values, real_y)


# In[256]:


# Display the corresponding confusion matrix for the cancer data (on the test set)

labels_cancer = LR_real.classes_
y_predictions = LR_real.predict(real_X.values)
cm_cancer = confusion_matrix(y_predictions, real_y)

print('Classification accuracy: {}%'.format(int(100*score_realdata)))
print("False Negative Rate: {}%".format(round(100*2/35, 2)))

plt.figure(figsize=(10,7))
plt.title('2 = Benign Tumor, 4 = Malignant Tumor')
sns.heatmap(cm_cancer, annot=True, xticklabels=labels_cancer, yticklabels=labels_cancer)
plt.xlabel('Logit C.F. Matrix Prediction')
plt.ylabel('Truth')


# In[372]:


# Let's try this analysis but with a random forest
# Let's use a RF with 20 estimators (so 20 trees) and same cross-validation sets
from sklearn.ensemble import RandomForestClassifier


def RFscoring(X_train, y_train, test_X, actual):
    forest = RandomForestClassifier(n_jobs=-1, n_estimators=30, min_samples_leaf=1, oob_score=False)
    forest.fit(X_train, y_train)
    score = forest.score(test_X, actual)
    return score

cv_scores_forest = []
for train_index, test_index in rkf.split(X.values):
    X_train = X.iloc[train_index, :].values
    y_train = y[train_index].values
    X_test = X.iloc[test_index, :].values
    y_truth = y[test_index].values
    fold_score = RFscoring(X_train, y_train, X_test, y_truth)
    cv_scores_forest.append(fold_score)


# In[373]:


np.mean(cv_scores_forest)


# In[374]:


forest_real = RandomForestClassifier(n_jobs=-1, n_estimators=30, min_samples_leaf=1, oob_score=False)
forest_real.fit(X.values, y.values)
forest_score_realdata = forest_real.score(real_X.values, real_y)


# In[375]:


from sklearn.metrics import precision_score, recall_score


# In[376]:


# Display the corresponding confusion matrix for the cancer data (on the test set)

labels_cancer = forest_real.classes_
y_predictions = forest_real.predict(real_X.values)
cm_cancer = confusion_matrix(y_predictions, real_y)

false_neg_rate = 1 - recall_score(real_y, y_predictions, average='macro')

print('Random Forest Classification accuracy: {}%'.format(int(100*forest_score_realdata)))
print("False Negative Rate: {}%".format(100*round(false_neg_rate,2)))

plt.figure(figsize=(10,7))
plt.title('2 = Benign Tumor, 4 = Malignant Tumor')
sns.heatmap(cm_cancer, annot=True, xticklabels=labels_cancer, yticklabels=labels_cancer)
plt.xlabel('Random Forest  Prediction')
plt.ylabel('Truth')


# In[310]:


forest_real.estimators_[0]


# In[316]:


forest_real.feature_importances_


# In[317]:


pd.DataFrame(forest_real.feature_importances_,
                                   index = X.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


# In[324]:


# Let's try to analyze the X.values matrix data with the sklearn decomp tool PCA
# Used for feature reduction and unsupervised patterns

from sklearn.decomposition import PCA

pca = PCA().fit(X.values)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[326]:


# With this PCA analysis, we can see that 4 features can caputre around 90% of the data! 
# It reflects the Random Forest feature_importances attribute that shows
# 4 main features with importance-percentages above 10%.
get_ipython().run_line_magic('pinfo2', 'PCA')


# In[355]:


full_data_no_labels = cancer_data_noID_removed.drop('Type', axis=1, inplace=False)
full_data_no_labels.values


# In[356]:


pca = PCA().fit(full_data_no_labels.values)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[358]:


# 2 components seems to obtain about 83% of the data. Let's try reducing

pca_2dim = PCA(n_components=2)  # project from 8 to 2 dimensions
projected = pca_2dim.fit_transform(full_data_no_labels.values)
print(full_data_no_labels.values.shape)
print(projected.shape)


# In[365]:


pd.DataFrame(pca_2dim.components_,columns=full_data_no_labels.columns,index = ['Component 1','Component 2'])


# In[366]:


projected


# In[ ]:





# In[ ]:





# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:22:17 2019

@author: Bjarne
"""
import pandas as pd
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer




def text_processing(tweet):
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)


def read_json(keyword):
    df = pd.read_json(f'./scraped_jsons/{keyword}_tweets.json', lines=False, encoding='utf-8')
    return df 
    
def most_common_words(keyword,k):
    df = read_json(keyword)
    words = []
    for t in df['text']:
        for w in text_processing(t):
            words.append(w)
    
    return list(pd.Series(words).value_counts().index)[:k]

keywords = ["SDAX","S&P 500","NASDAQ 100","STOXX 50","EURO STOXX 50","FTSE 100","CAC 40","NIKKEI 225","Hang Seng"]
keywords_url = [k.replace(" ","%20").replace("&","%26").replace("-","%2D") for k in keywords]


#%% Find most common words in unflagged tweets
unflagged_common_words = most_common_words("unflagged",200)
selected_keywords = []

for k in keywords_url:
    fond_common_keywords = most_common_words(k,10)
    
    # Build difference between fond commons and unflagged commons 
    filtered_keywords = list(set(fond_common_keywords).difference(unflagged_common_words))
    
    # Append Keywords to selected keywords
    for k in filtered_keywords[:3]:
        selected_keywords.append(k)
        

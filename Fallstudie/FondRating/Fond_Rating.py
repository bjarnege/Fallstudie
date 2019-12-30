# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:22:43 2019

@author: Bjarne
"""
#%% Imports
import pandas as pd
import urllib.request, json 
import requests
from textblob import TextBlob
import re
import joblib
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

#%% Load Pretrained Models
sent_classifier = '../Sentiment_Classifier/sentiment_classifier.sav'
sentiment_model = joblib.load(sent_classifier)

fond_classifier = '../Fond_Classifier/fond_classifier.sav'
fond_model = joblib.load(fond_classifier)


#%% Load new Data from DB
with urllib.request.urlopen("http://h2655330.stratoserver.net:5431/get/actual_data") as url:
    data = json.loads(url.read().decode())
    
data = data['items']
tweets = pd.DataFrame.from_dict(data)

#%% Predict Sentiment and Stock 
pred_sent = [p[2]-p[0] for p in sentiment_model.predict_proba(tweets['text'])]
#%% Predict Stock
pred_fonds = fond_model.predict_proba(tweets['text'])

#%% Create DataFrame
df = pd.DataFrame(pred_fonds,columns=list(fond_model.classes_))
df['Sentiment'] = pred_sent

#%%Determine allover Sentiment
def weighted_sentiment(fond,df):
    return (df[fond] * df['Sentiment']).sum() / df[fond].sum()

def risk_score(fond,df):
    # Dertermine classes where fond got the highest prediction and return Std 
    return df[df.drop("Sentiment",axis=1).T.idxmax() == fond]['Sentiment'].std()
    
#%% Insert Keyfigures into db
fonds = df.columns[:-1]
for f in fonds:
    weighted_sent = weighted_sentiment(f,df)
    risk_ = risk_score(f,df)
    sum_class = df[df.drop("Sentiment",axis=1).T.idxmax() == f].shape[0]

    payload = {'indexfonds': f, 'avg_sentiment': weighted_sent,'risk': risk_, 'sum_classifier': sum_class}
    requests.post("http://h2655330.stratoserver.net:5431/post/prediction", data=payload)
    
    
    

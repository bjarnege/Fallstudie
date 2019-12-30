# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:50:39 2019

@author: Bjarne
"""
import pandas as pd
import joblib
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, accuracy_score
import urllib.request, json 


#%% Accessing Training-Data
with urllib.request.urlopen("http://h2655330.stratoserver.net:5431/get/training_data") as url:
    data = json.loads(url.read().decode())
    
data = data['items']
    
tweets = pd.DataFrame.from_dict(data)#,orient="records", lines=True, encoding='utf-8')
tweets = tweets[tweets["keyword"] != "Test keyword"]
tweets = tweets[tweets["keyword"] == "test5"]


#%% Create Batches to determine accuracy later

# Shuffle DataFrame
tweets = tweets.sample(frac=1)

# Get Train and Test
Train = tweets.iloc[:tweets.shape[0]-500,:]
Test = tweets.iloc[tweets.shape[0]-500:,:]


#%% Data cleaning

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


#%% Create Model
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(Train['text'],Train['indexfonds'])


#%% Determine accuarcy
y_pred = pipeline.predict(Test['text'])
acc = accuracy_score(y_pred,Test['indexfonds'])


#%% Export Model
# save the model to disk
filename = './fond_classifier.sav'
joblib.dump(pipeline, filename)

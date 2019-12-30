# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:21:25 2019

@author: Bjarne
"""
import os
import requests
import datetime
import pandas as pd

history_dt = (datetime.datetime.today() - datetime.timedelta(weeks=24)).date()

def scrape_keyword(begindate,keyword):
    
    try:
        os.mkdir("./scraped_jsons")
    except:
        pass
    
    os.system(f"twitterscraper {keyword} --limit 1000 --output=./scraped_jsons/{keyword}_tweets.json --lang=en -bd {str(begindate)}" )

def read_json(keyword):
    df = pd.read_json(f'./scraped_jsons/{keyword}_tweets.json', lines=False, encoding='utf-8')
    return df 
    
#%% Keywords scapren und speichern in lokalem Verzeich6nis scraped_jsons

#keywords = ["SDAX","S&P 500","NASDAQ 100","STOXX 50","EURO STOXX 50","FTSE 100","SMI","SPI","CAC 40","NIKKEI 225","Hang Seng"
#,"IBEX 35","ATX","OMXS30","AEX","NYSE US 100","BUX","WIG 20","SOFIX","RTS","NSE 20","TA-100","EGX30","GSE","BOVESPA","IGPA"
#,"Merval 25","KSE 100","ISE 100","SENSEX","TEPIX","KOSPI","TSX 60","CSI 300","OSEBX"]

keywords = ["SDAX","S&P 500","NASDAQ 100","STOXX 50","EURO STOXX 50","FTSE 100","CAC 40","NIKKEI 225","Hang Seng"]


# Keywords in URL-Format umwandeln
keywords_url = [k.replace(" ","%20").replace("&","%26").replace("-","%2D") for k in keywords]
# News_Scrapen und anlegen
for k in keywords_url:
    scrape_keyword(history_dt,k)
    
# News ohne Keyword scrapen (Neutrale Daten)
os.system(f'twitterscraper "economy OR politics OR fund OR stock" --limit 1000 --output=./scraped_jsons/unflagged_tweets.json --lang=en')
    
#%% Aufbereiten der gescrapten Dataframes und einfügen in die DB
keywords.append("unflagged")
keywords_url.append("unflagged")

for k_url,k_raw in zip(keywords_url,keywords):
    df = read_json(k_url)
    df["indexfonds"] = k_raw
    df['keyword'] = ["".join([x+"," for x in X])[:-1] for X in df["hashtags"]]
    df = df[["text","keyword","indexfonds","timestamp","tweet_url"]].drop_duplicates(subset=['text'], keep='last')

    for line in df.values:
        payload = {'text': line[0], 'keyword': line[1], 'indexfonds': line[2], 'date': line[3], 'link': f"https://twitter.com{line[4]}"}
        requests.post("http://h2655330.stratoserver.net:5431/post/training_data", data=payload)
    
#%% 

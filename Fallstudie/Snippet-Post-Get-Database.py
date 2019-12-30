import datetime, requests

#Tabelle 1
text = 'Test Text'
keyword = 'Test keyword'
indexfonds = "Test Indexfonds"
#date = datetime.datetime(2009, 5, 5)
date = datetime.datetime.now()
link = 'Test link'
payload = {'text': text, 'keyword': keyword, 'indexfonds': indexfonds, 'date': date, 'link': "link"}
r = requests.post("http://h2655330.stratoserver.net:5431/post/training_data", data=payload)

# http://h2655330.stratoserver.net:5431/get/training_data
# http://h2655330.stratoserver.net:5431/get/training_data?date=2009-05-05

#Tabelle 2
text = 'Test Text'
keyword = 'Test keyword'
date = datetime.datetime.now()
link = 'Test link'
payload = {'text': text, 'keyword': keyword,'date': date, 'link': "link"}
r = requests.post("http://h2655330.stratoserver.net:5431/post/actual_data", data=payload)

# http://h2655330.stratoserver.net:5431/get/actual_data
# http://h2655330.stratoserver.net:5431/get/actual_data?date=2019-12-27

#Tabelle 3
indexfonds = 'Test indexfonds'
avg_sentiment = 5
risk = 75
sum_classifier = 100
payload = {'indexfonds': indexfonds, 'avg_sentiment': avg_sentiment,'risk': risk, 'sum_classifier': sum_classifier}
r = requests.post("http://h2655330.stratoserver.net:5431/post/prediction", data=payload)

# http://h2655330.stratoserver.net:5431/get/predictions
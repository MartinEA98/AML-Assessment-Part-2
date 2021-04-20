import numpy as np
import nltk
import random
import operator
import sklearn
import string
import os
import sys
from random import randrange
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

#Here you can specifiy a random data instance to train the ML model. If not arguement is given it will create a rando instance.
#This allows you to repeat tests across a consistant data space.
if len(sys.argv) < 2:
    rand = randrange(50)
else:
    rand = sys.argv[1]


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#Test model: This code simply takes the training model and the data it's to be tested on, in the form [X_test, Y_test],
# and reports back the macro averaged scores across precision, recall, f1-score and accuracy.
def test_model(model, testdata):
    print("Beginning testing...")
    X_test = testdata[0]
    Y_test = testdata[1]

    predictions = model.predict(X_test)

    print(classification_report(Y_test, predictions))
    precision=precision_score(Y_test, predictions, average='macro')
    recall=recall_score(Y_test, predictions, average='macro')
    f1=f1_score(Y_test, predictions, average='macro')
    accuracy=accuracy_score(Y_test, predictions)

    print ("Precision: "+str(round(precision,5)))
    print ("Recall: "+str(round(recall,5)))
    print ("F1-Score: "+str(round(f1,5)))
    print ("Accuracy: "+str(round(accuracy,5)))

    return precision, recall, f1, accuracy


#RETRIEVING DATA---------------------------------------------------
#This 'datafile_fict' specifices how many instances of each article type will be retrieved from their respective
# folders. This can be changed if more data is included, or if some is removed.
datafile_dict = {}
datafile_dict["business"] = 510
datafile_dict["entertainment"] = 386
datafile_dict["politics"] = 417
datafile_dict["sport"] = 511
datafile_dict["tech"] = 401

keys = ["business", "entertainment","politics", "sport", "tech"]

data_business_raw = []
data_entertainment_raw = []
data_politics_raw = []
data_sport_raw = []
data_tech_raw = []

#Retrieve current working directory
cwd = str(os.getcwd())

#This for loop goes through each key in the 'datafile_dict' above and creates a list of paths
# from which data will be retrieved and placed into seperate lists (one per category).
for key in datafile_dict:
    paths = []
    num_data = datafile_dict[key]
    count = 1
    for i in range(num_data):
        if count < 10:
            path = cwd + "\\data\\bbc\\" + key +"\\00" + str(count) + ".txt"
        elif count < 100:
            path = cwd + "\\data\\bbc\\" + key +"\\0" + str(count) + ".txt"
        else:
            path = cwd + "\\data\\bbc\\" + key +"\\" + str(count) + ".txt"
        count += 1
        paths.append(path)  
    print("Grabbing " + key +" data...")
    data_list_raw = []
    for path in paths:
        with open(path, encoding="utf8", errors='ignore') as f:
            data_list_raw.append(f.readlines())

    if key == "business": data_business_raw = data_list_raw
    elif key == "entertainment": data_entertainment_raw = data_list_raw
    elif key == "politics": data_politics_raw = data_list_raw
    elif key == "sport": data_sport_raw = data_list_raw
    elif key == "tech": data_tech_raw = data_list_raw
  
print("Number of business articles: ", len(data_business_raw))
print("Number of entertainment articles: ", len(data_entertainment_raw))
print("Number of politics articles: ", len(data_politics_raw))
print("Number of sport articles: ", len(data_sport_raw))
print("Number of tech articles: ", len(data_tech_raw))
print("Deleting duplicated data...")
data_business = []
data_entertainment = []
data_politics = []
data_sport = []
data_tech = []

#REMOVE REPLICATED DATA
#The dataset contains some replicated data, thus needs to be checked 
# and duplicates removed.

for i, article1 in enumerate(data_business_raw):
  for j, article2 in enumerate(data_business_raw):
    if i == j: continue
    elif article1 == article2:
      count += 1
      del data_business_raw[j]
  

for i, article1 in enumerate(data_entertainment_raw):
  for j, article2 in enumerate(data_entertainment_raw):
    if i == j: continue
    elif article1 == article2:
      count += 1
      del data_entertainment_raw[j]

for i, article1 in enumerate(data_politics_raw):
  for j, article2 in enumerate(data_politics_raw):
    if i == j: continue
    elif article1 == article2:
      count += 1
      del data_politics_raw[j]

for i, article1 in enumerate(data_sport_raw):
  for j, article2 in enumerate(data_sport_raw):
    if i == j: continue
    elif article1 == article2:
      count += 1
      del data_sport_raw[j]

for i, article1 in enumerate(data_tech_raw):
  for j, article2 in enumerate(data_tech_raw):
    if i == j: continue
    elif article1 == article2:
      count += 1
      del data_tech_raw[j]


print("Data grabbed.\n")
print("Number of business articles: ", len(data_business_raw))
print("Number of entertainment articles: ", len(data_entertainment_raw))
print("Number of politics articles: ", len(data_politics_raw))
print("Number of sport articles: ", len(data_sport_raw))
print("Number of tech articles: ", len(data_tech_raw))
print("Number of deleted duplicate articles: ", count,"\n")

#Create labels to be used in training. Will take the form, [num_arricles, 1], where each instance 
# will be 0,1,2,3 or 4 corresponding to business, entertaiment, politics, sport and tech articles 
# respectively
labels = []

for i in range(len(data_business_raw)):
  labels.append(0)

for i in range(len(data_entertainment_raw)):
  labels.append(1)

for i in range(len(data_politics_raw)):
  labels.append(2)

for i in range(len(data_sport_raw)):
  labels.append(3)

for i in range(len(data_tech_raw)):
  labels.append(4)

labels = np.asarray(labels)

all_data_raw = []
all_data_raw = (data_business_raw + data_entertainment_raw + data_politics_raw + data_sport_raw + data_tech_raw)

#Download stop-words for filtering tokens later. 
stopwords=set(nltk.corpus.stopwords.words('english'))
#Custom stop words may be added as below:
stopwords.add(".")
stopwords.add(",")
stopwords.add("``")
stopwords.add("''")
stopwords.add("'s")
stopwords.add("-")
stopwords.add("(")
stopwords.add(")")
stopwords.add(":")
stopwords.add("'")
stopwords.add("''")
stopwords.add("!")

print("Beginning feature creation...")

#CREATE PAIRS FEATURE
all_pairs_tokens = []
lemmatizer = nltk.stem.WordNetLemmatizer() #Create lemmatizer object to lemmatize data

for article in all_data_raw: #loop thourgh each article
  list_tokens = [] #create empty list of tokens
  for line in article: #loop through each line in article
    sentence_split = nltk.tokenize.sent_tokenize(str(line)) #split line into sentences
    for sentence in sentence_split: #loop through each sentence
      #1) remove punctuation from sentence 2) split into individual tokens
      list_tokens_sentence = nltk.tokenize.word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation)))
      for i, token in enumerate(list_tokens_sentence): #loop through all tokens from previous sentence
        if i >= 1: #if not the first token in list (this would cause indexing error)
          pair = list_tokens_sentence[i-1] + " " + list_tokens_sentence[i]  #Add to list each pair of tokens
          list_tokens.append(lemmatizer.lemmatize(pair.lower())) #1) make pair lower case and lemmatize
  all_pairs_tokens.append(list_tokens) #append all tokens from article to 'all_pairs_tokens'

mf_all_pairs = None #Variable to control max number of most freq. pairs included in vector.
kbest_all_pairs = 1000 #Variable to control max K-best dimensionality
count_vector_all_pairs = CountVectorizer(input='content',decode_error='ignore',
                               stop_words=stopwords, 
                               max_features=mf_all_pairs,
                               analyzer=lambda x: x) #Define vectorizer object
feature_all_pairs = count_vector_all_pairs.fit_transform(all_pairs_tokens) #Using on 'all_pairs_tokens' to get back numerical vector for training.
tfidf_transformer = TfidfTransformer().fit(feature_all_pairs) #Define and fit 'tfidf' normaliser to data, by default this is using the 'l2' norm
features_all_pairs_final = tfidf_transformer.transform(feature_all_pairs).toarray() #normalise data
features_all_pairs_final_kb = SelectKBest(chi2, k=200).fit_transform(features_all_pairs_final, labels) #Perform dimensionality reduction with SelectKBest
#--------------------------------------------------------------------

#CREATE LENGTH FEATURE--------------------------------------------
feature_length = []

for article in all_data_raw:#loop thourgh each article
  list_tokens = []#create empty list of tokens
  for line in article:#loop through each line in article
    sentence_split = nltk.tokenize.sent_tokenize(str(line))#split line into sentences
    for sentence in sentence_split:#loop through each sentence
      #1) remove punctuation from sentence 2) split into individual tokens
      list_tokens_sentence = nltk.tokenize.word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation)))
      for token in list_tokens_sentence: #loop through all tokens from previous sentence
        list_tokens.append(token.lower()) #append tokens to list of temporary tokens
  feature_length.append(len(list_tokens)) #count length of article and append number of 'feature_length'

feature_length = np.asarray(feature_length) #convert to array
scaler = MaxAbsScaler().fit(feature_length.reshape(-1,1)) #fit MaxAbsScaler scaler
feature_length_final = scaler.transform(feature_length.reshape(-1,1)) #normalise data
#--------------------------------------------------------------


#CREATE SINGLES FEATURE----------------------------------------
#Process is identical to PAIRS of features
all_tokens = []
lemmatizer = nltk.stem.WordNetLemmatizer()

for article in all_data_raw:
  list_tokens = []
  for line in article:
    sentence_split = nltk.tokenize.sent_tokenize(str(line))
    for sentence in sentence_split:
      list_tokens_sentence = nltk.tokenize.word_tokenize(sentence.translate(str.maketrans('', '', string.punctuation))) #Strips away punctation
      for token in list_tokens_sentence:
        list_tokens.append(lemmatizer.lemmatize(token.lower()))
  all_tokens.append(list_tokens)

count_vector = CountVectorizer(input='content',decode_error='ignore',
                               stop_words=stopwords, 
                               max_features=None,
                               analyzer=lambda x: x)
feature = count_vector.fit_transform(all_tokens)
tfidf_transformer = TfidfTransformer().fit(feature) #By default this is using the 'l2' norm
feature_final = tfidf_transformer.transform(feature).toarray()
feature_final_kb = SelectKBest(chi2, k=1200).fit_transform(feature_final, labels)
#------------------------------------------------------------------

print("Features created.\n")


#MODEL CREATION---------------------------------------------------
if sys.argv != None:
    rand = randrange(50)
else:
    rand = sys.argv
#Note, random state '26' was used to produce the results in the report
#Append features to single vector
feature_second = np.append(feature_final_kb, features_all_pairs_final_kb, axis=1)
feature_second = np.append(feature_second, feature_length_final, axis=1)

#Create 70/15/15 training, dev, test sets
X_train, X_test, Y_train, Y_test = train_test_split(feature_second, labels, test_size=0.3, random_state=rand)
X_test, X_dev, Y_test, Y_dev = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand)
print("Beginning model training...")
model = sklearn.svm.SVC(kernel='rbf', gamma='scale') #define model
model.fit(X_train, Y_train) #Train model
print("Model trained.\n")
print("Below is the test results using the development set:")
p, r, f, a = test_model(model, [X_dev, Y_dev])
print("Below is the test results using the unseen test set:")
p, r, f, a = test_model(model, [X_test, Y_test])
#-----------------------------------------------------------------






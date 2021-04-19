# AML-Assessment-Part-2

Welcome! This is the GitHub repository holding the code used to train a Machine Learning (ML) model to classify news articles based on the 'BBC' dataset. 

To run the code for yourself, simply download both the 'train_ml_model.py' file and the data in the 'data.zip' file. Place these in the same directory and run with the following command:

python train_ml_model.py

The program will train on a random split of the data provided and will report both it's performance on the development and test sets.

An explaination of the steps implemented by my code as as follows:

1) Data retrieval

The data is retrieved from the fil system and placed into one large list. The labels required for training later on are also created here, taking values of [0,1,2,3,4] to mean the article is of class [business, entertainment, politics, sport, tech] respectively. Each data point (article) takes the form of a series of lines, with each line consisting of one of more sentences, forming a single string.

2) Feature Creation

Here I will give an overview of the principles behind each of the features within the code, a detailed annotated breakdown can be found within the commend of the code itself.

Feature 1: This feature is a simple word occurrence frequency matrix. The purpose of the code is to take an article and break it down to it's individual words (punctation, cases and more are discarded in the process for simplification) and creating a vector corresponding to the number of times a given word appears within the article. Within the code there is a short script to breakdown the article to it's individual words, but the vectorization is handled by the sklearn "CountVectorizer" method, the documentation for which can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html. The resultant vector can be read by our ML model and used for training.

During the processing of our articles into words we use a technique known as 'lemmatization'. This is a form of stemming in which various inflections of a singular word are instead reduced to a basic version of the original word. Whilst we will lose some information in the text which could have aided us in our ML efforts, it also greatly simplifies our data and allows for easier and faster training. Imagine, we can take 7 inflections of a word and have it reduced to a single number represented in our CountVector! A detailed explaination of this concept, and the exact method we use in the code to achieve this can be found here: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/

We will also make use of stop-words. These are a set of common words and particles in English which (for our purposes) are of no use. These include words such as 'him', 'it', 'and', and 'the'. This once again simplifies our data and makes learning from it easier.

Next we normalise our data. We do this using the sklearn 'TfidfTransformer' where the 'tfidf' means term-frequency times inverse document-frequency. This is a very useful data normalisation technique which keeps our data sparse, which is very important for our count vector. a detailed explaination and the documentation for this can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html

Finally, we use sklearns 'SelectKBest' to perform dimensionality reduction. This method allows us to select the most important features from our overall set so we may discard the rest. This decreases our overal training time and increasing performance futher. A detailed explaination of this method can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

Feature 2: This is conceptually almost identical to feature 1, however instead of individual words, we simply use word pairs. These can encode very useful information in our text and can boost the performance of our final algorithm. The lemmatization, normalisation and dimensionality reduction remains the same as for feature 1 also.

Feature 3: This feature is simply the average length of an article in words. We normalise it with the same 'tfidf' scheme as before.

Finally, we append all the features into a single vector ready for training.

Model Training:

The model to be trainied is an sklearn support-vector classifier.





# AML-Assessment-Part-2

Welcome! This is the GitHub repository holding the code used to train a Machine Learning (ML) model to classify news articles based on the 'BBC' dataset. The steps implemented by my code as as follows:

1) Data retrieval

The data is retrieved from the fil system and placed into one large list. The labels required for training later on are also created here, taking values of [0,1,2,3,4] to mean the article is of class [business, entertainment, politics, sport, tech] respectively. Each data point (article) takes the form of a series of lines, with each line consisting of one of more sentences, forming a single string.

2) Feature Creation

Here I will give an overview of the principles behind each of the features within the code, a detailed annotated breakdown can be found within the commend of the code itself.

Feature 1: This feature is a simple word occurrence frequency matrix. The purpose of the code is to take an article and break it down to it's individual words (punctation, cases and more are discarded in the process for simplification) and creating a vector corresponding to the number of times a given word appears within the article. Within the code there is a short script to breakdown the article to it's individual words, but the vectorization is handled by the sklearn "CountVectorizer" method, the documentation for which can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html. The resultant vector can be read by our ML model and used for training.

During the processing of our articles into words we use a technique known as 'lemmatization'. This is a form of stemming in which various inflections of a singular word are instead reduced to a basic version of the original word. Whilst we will lose some information in the text which could have aided us in our ML efforts, it also greatly simplifies our data and allows for easier and faster training. Imagine, we can take 7 inflections of a word and have it reduced to a single number represented in our CountVector! A detailed explaination of this concept, and the exact method we use in the code to achieve this can be found here: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/





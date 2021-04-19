# AML-Assessment-Part-2

Welcome! This is the GitHub repository holding the code used to train a Machine Learning (ML) model to classify news articles based on the 'BBC' dataset. The steps implemented by my code as as follows:

1) Data retrieval

The data is retrieved from the fil system and placed into one large list. The labels required for training later on are also created here, taking values of [0,1,2,3,4] to mean the article is of class [business, entertainment, politics, sport, tech] respectively. Each data point (article) takes the form of a series of lines, with each line consisting of one of more sentences, forming a single string.

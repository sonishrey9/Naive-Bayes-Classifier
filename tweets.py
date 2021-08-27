# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# %%
tweets = pd.read_csv("Disaster_tweets_NB.csv")
tweets


# %%
tweets.info()


# %%
for i in tweets.target:

    if i == 1 :
        tweets["target"] = tweets["target"].replace([1], "real tweet")
    elif i == 0:
        tweets["target"] = tweets["target"].replace([0], "Fake tweet")


# %%
tweets.head(50)


# %%
# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open("stopwords_en.txt","r") as sw:
    stop_words = sw.read()


# %%
stop_words = stop_words.split("\n")


# %%
stop_words


# %%
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))


# %%
tweets.text = tweets.text.apply(cleaning_text)
tweets.text


# %%
# removing empty rows
tweets = tweets.loc[tweets.text != " ",:]


# %%
# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

tweets_train, tweets_test = train_test_split(tweets, test_size = 0.2)


# %%
# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]


# %%
# Defining the preparation of email texts into word count matrix format - Bag of Words
tweets_bow = CountVectorizer(analyzer = split_into_words).fit(tweets.text)
tweets_bow


# %%
# Defining BOW for all messages
all_tweets_matrix = tweets_bow.transform(tweets.text)
all_tweets_matrix


# %%
# For training messages
train_tweets_matrix = tweets_bow.transform(tweets_train.text)
train_tweets_matrix


# %%
# For testing messages
test_tweets_matrix = tweets_bow.transform(tweets_test.text)


# %%
# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tweets_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tweets_matrix)
train_tfidf.shape # (row, column)


# %%
# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweets_matrix)
test_tfidf.shape #  (row, column)

# %% [markdown]
# # **Preparing a naive bayes model on training data set**

# %%
from sklearn.naive_bayes import MultinomialNB as MB


# %%
tweets_train


# %%
# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweets_train.target)


# %%
# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweets_test.target)
accuracy_test_m


# %%
test_pred_m


# %%
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweets_test.target) 


# %%
pd.crosstab(test_pred_m, tweets_test.target)


# %%
# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweets_train.target)
accuracy_train_m


# %%
train_pred_m

# %% [markdown]
# ## Multinomial Naive Bayes changing default alpha for laplace smoothing
# ## if alpha = 0 then no smoothing is applied and the default alpha parameter is 
# ##  the smoothing process mainly solves the emergence of zero probability problem in the dataset.

# %%
classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, tweets_train.target)


# %%
# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweets_test.target)
accuracy_test_lap


# %%
test_pred_lap


# %%
from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, tweets_test.target) 


# %%
pd.crosstab(test_pred_lap, tweets_test.target)


# %%
# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweets_train.target)
accuracy_train_lap


# %%
train_pred_lap


# %%
train_pred_lap


# %%




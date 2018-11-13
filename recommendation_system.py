import pandas as pd
import numpy as np

# url: https://www.datacamp.com/community/tutorials/recommender-systems-python

# Content-Based Recommender in Python
# Plot Description Based Recommender
metadata = pd.read_csv('movies_metadata.csv')
print(metadata)

# Since you are trying to build a clone of IMDB's Top 250, you will use its weighted rating formula as your metric/score. Mathematically, it is represented as follows:
#
# Weighted Rating (WR) = (v/(v+m).R)+(m/(v+m).C)
# where,
#
# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report

# Calculate C
C = metadata['vote_average'].mean()
print(C)

#Next, let's calculate the number of votes, m, received by a movie in the 90th percentile. The pandas library makes this task extremely trivial using the .quantile() method of a pandas Series:

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)

#Next, you can filter the movies that qualify for the chart, based on their vote counts:

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(q_movies.shape)

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
# print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(50))
#
# print(metadata['overview'].head())

# plot overview based movie recommendation

# In its current form, it is not possible to compute the similarity between any two overviews.
# To do this, you need to compute the word vectors of each overview or document,
# as it will be called from now on.
#
# You will compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document.
# This will give you a matrix where each column represents a word in the overview vocabulary
# (all the words that appear in at least one document) and each column represents a movie, as before.
#
# In its essence, the TF-IDF score is the frequency of a word occurring in a document,
# down-weighted by the number of documents in which it occurs.
# This is done to reduce the importance of words that occur frequently in plot overviews and therefore,
# their significance in computing the final similarity score.

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)
# print(tfidf_matrix)


# You see that over 75,000 different words were used to describe the 45,000 movies in your dataset.
#
# With this matrix in hand, you can now compute a similarity score.
# There are several candidates for this; such as the euclidean, the Pearson and the cosine similarity scores.
# Again, there is no right answer to which score is the best.
# Different scores work well in different scenarios and it is often a good idea to experiment with different metrics.
#
# You will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies.
# You use the cosine similarity score since it is independent of magnitude
# and is relatively easy and fast to calculate (especially when used in conjunction with TF-IDF scores,
# which will be explained later). Mathematically, it is defined as follows:
#
# cosine(x,y)=x.y⊺||x||.||y||
# Since you have used the TF-IDF vectorizer,
# calculating the dot product will directly give you the cosine similarity score.
# Therefore, you will use sklearn's linear_kernel() instead of cosine_similarities() since it is faster.


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

#You're going to define a function that takes in a movie title as an input
# and outputs a list of the 10 most similar movies. Firstly,
# for this, you need a reverse mapping of movie titles and DataFrame indices.
# In other words, you need a mechanism to identify the index of a movie in your metadata DataFrame, given its title.

#Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
print(indices)
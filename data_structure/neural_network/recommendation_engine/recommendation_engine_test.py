import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

df = pd.read_csv("movie.csv")
#pd.set_option('expand_frame_repr', False)

df = df[["director_name", "genres", "actor_1_name", "movie_title", "imdb_score", "plot_keywords"]]
print(df)

# content based recommendation engine

# plot_keywords only
# tf-idf vector for plot_keywords

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#print(df['plot_keywords'].isnull().sum())
#print(df[df['plot_keywords'].isnull()])

df.dropna(how='any',axis=0,inplace=True)

def replace_str(x):
    return x.replace("|"," ")


df['plot_keywords'] = df.apply(lambda row:replace_str(row['plot_keywords']),axis=1)
#print(df)

def strip(x):
    return x.strip()

df['movie_title'] = df.apply(lambda row:strip(row['movie_title']),axis=1)
#print(df)

# tf-idf with n-gram
vectorizer = TfidfVectorizer(ngram_range=(1,1))
# tokenize and build vocab
#print("list of strings",list(df['plot_keywords'].values))
vectorizer.fit(list(df['plot_keywords'].values))
# summarize
#print(vectorizer.vocabulary_)


movie_title = input("input movie title(Case Sensitive)")
print("selected title: ",movie_title)
df_movie = df[df['movie_title'] == movie_title]
pd.set_option('expand_frame_repr', False)
print(df_movie)

print("get selected movie: ",df_movie.loc[:,"plot_keywords"].values)
vector = vectorizer.transform(df_movie.loc[:,"plot_keywords"])
#print("vector shape: ",vector.shape)
#print(vector.toarray())

from sklearn.metrics.pairwise import cosine_similarity


def cosine_put(x):
    vector_row = vectorizer.transform([x])
    #print("vector row shape: ", vector_row.shape)
    #print(vector_row.toarray())
    cosine_score = cosine_similarity(vector,vector_row)
    #print(cosine_score)
    return cosine_score[0][0]

df['cosine_score'] = df.apply(lambda row:cosine_put(row['plot_keywords']),axis=1)

#print(df['cosine_score'])

df.sort_values("cosine_score",ascending=False,inplace=True)
print(df[['movie_title','cosine_score','plot_keywords']].head(10))
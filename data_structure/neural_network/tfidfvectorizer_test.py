from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
documents = ["this is omi","omi is a good boy","omi is my name","this is a cat"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(documents)
# summarize
print(vectorizer.vocabulary_)

vector1 = vectorizer.transform(["omi is a good cat"])
print(vector1.toarray())
vector2 = vectorizer.transform(["cat is a good boy"])
print(vector2.toarray())

from sklearn.metrics.pairwise import cosine_similarity

cosine_score = cosine_similarity(vector1,vector2)
print(cosine_score[0][0])
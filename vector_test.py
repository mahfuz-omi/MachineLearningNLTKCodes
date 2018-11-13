from sklearn.feature_extraction import text

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

stemmer = PorterStemmer()

text1 = 'i love her'
text2 = 'she is loved by me'
text3 = 'she loves me'

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

documentList = [text1,text2,text3]
vectorizer = text.CountVectorizer(tokenizer=tokenize)
results = vectorizer.fit_transform(documentList)
print(vectorizer.vocabulary_)
print(results.toarray())





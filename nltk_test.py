from sklearn.feature_extraction import text
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


vocab = ['Sam loves swimming so he swims all the time']
vect = text.CountVectorizer(tokenizer=tokenize, stop_words='english')
vec = vect.fit(vocab)
sentence1 = vec.transform(['George loves swimming too!'])
print(vec.get_feature_names())
print(sentence1.toarray())

# word tokenization
from nltk.tokenize import word_tokenize

text = "i love to eat and i am eating"
tokenized_word_list = word_tokenize(text)
print(tokenized_word_list)


# stemming
from nltk.stem import SnowballStemmer
snow = SnowballStemmer('english')

print(snow.stem('getting')) # get

print(snow.stem('rabbits'))  # rabbit

print(snow.stem('xyzing'))  # xyze - it even works on non words!

print(snow.stem('quickly'))  # quick

print(snow.stem('slowly'))  # slowli

from nltk.stem import PorterStemmer
porter = PorterStemmer()

print(porter.stem('getting')) # get

print(porter.stem('rabbits'))  # rabbit

print(porter.stem('xyzing'))  # xyze - it even works on non words!

print(porter.stem('quickly'))  # quickli

print(porter.stem('slowly'))  # slowli

# so,snowball works better than porter

# lemmatization-
import nltk
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer

lematizer = WordNetLemmatizer()
lematizated_word_list = [lematizer.lemmatize(word) for word in tokenized_word_list]
print(lematizated_word_list)


# removing stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
print(stop_words)

stop_words_filtered = []
for word in tokenized_word_list:
    if word not in stop_words:
        stop_words_filtered.append(word)


print(stop_words_filtered)


# word vector

# 1 hot scheme (not distributed) == tf-idf == count vector
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
documents = ["this is omi","omi is a good boy","omi is my name","this is a cat"]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(documents)
# summarize
print(vectorizer.vocabulary_)
# encode document
# vector = vectorizer.transform(text)
# # summarize encoded vector
# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
x = vectorizer.transform(["omi","good","name","omi is "])
print(type(x))
#<class 'scipy.sparse.csr.csr_matrix'>


#print(x.toarray())
# [[0 0 0 0 0 0 1 0]
#  [0 0 1 0 0 0 0 0]
#  [0 0 0 0 0 1 0 0]]

# tf-idf with n-gram
vectorizer = TfidfVectorizer(ngram_range=(1,1))
# tokenize and build vocab
vectorizer.fit(documents)
# summarize
print("tf-idf")
print(vectorizer.vocabulary_)
# encode document
# vector = vectorizer.transform(text)
# # summarize encoded vector
# print(vector.shape)
# print(type(vector))
# print(vector.toarray())
x = vectorizer.transform(["omi","good","name","omi is"])
print(type(x))
#<class 'scipy.sparse.csr.csr_matrix'>

#print(x.toarray())

# Hash Vectorizer
vectorizer = HashingVectorizer(n_features=5)
# tokenize and build vocab
vectorizer.fit(documents)
x = vectorizer.transform(["omi omi","good","name","omi is"])
print(x.toarray())




import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

from nltk.corpus import stopwords
import string

stopwords_english = stopwords.words('english')


# feature extractor function
def bag_of_words(words):
    words_clean = []
    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)
    words_dictionary = dict([word, True] for word in words_clean)
    return words_dictionary

# Create Feature Set

# We use the bag-of-words feature and tag each review with its respective category as positive or negative.

# positive reviews feature set
pos_reviews_set = []
for words in pos_reviews:
    pos_reviews_set.append((bag_of_words(words), 'pos'))

# negative reviews feature set
neg_reviews_set = []
for words in neg_reviews:
    neg_reviews_set.append((bag_of_words(words), 'neg'))

# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
from random import shuffle

shuffle(pos_reviews_set)
shuffle(neg_reviews_set)

test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

# Training Classifier and Calculating Accuracy

from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_set)

# Testing Classifier with Custom Review

from nltk.tokenize import word_tokenize

custom_review = input("please input movie review")
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)
print(classifier.classify(custom_review_set))
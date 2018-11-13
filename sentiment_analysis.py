# https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184

# http://blog.chapagain.com.np/python-nltk-sentiment-analysis-on-movie-reviews-natural-language-processing-nlp/
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

print(movie_reviews)
# <CategorizedPlaintextCorpusReader in '.../corpora/movie_reviews'

print("fields: ",movie_reviews.fileids())
# fields:  ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', 'neg/cv002_17424.txt', 'neg/cv003_12683.txt', 'neg/cv004_12641.txt', 'neg/cv005_29357.txt', 'neg/cv006_17022.txt']

# Total reviews
print (len(movie_reviews.fileids())) # Output: 2000

# Review categories
print(movie_reviews.categories())  # Output: [u'neg', u'pos']

# Total positive reviews
print(len(movie_reviews.fileids('pos')))  # Output: 1000

# Total negative reviews
print(len(movie_reviews.fileids('neg')))  # Output: 1000

# movie_review.fileids(category) returns the fileids with that category
# movie_review.fileids() returns all fileids with their respective categories
# movie_review.words(fileid) returns the corresponsding words in that file

# processing datas
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        # documents.append((list(movie_reviews.words(fileid)), category))
        documents.append((movie_reviews.words(fileid), category))

print(len(documents))  # Output: 2000

print(documents)
# [(['plot', ':', 'two', 'teen', 'couples', 'go', 'to', ...], 'neg'), (['the', 'happy', 'bastard', "'", 's', 'quick', 'movie', ...], 'neg')]

# print first tuple
print (documents[0])
# (['plot', ':', 'two', 'teen', 'couples', 'go', 'to', ...], 'neg')

# shuffle the document list
from random import shuffle
shuffle(documents)

# feature extraction
# Fetch all words from the movie reviews corpus

all_words = [word.lower() for word in movie_reviews.words()]

# print first 10 words
print(all_words[:10])

# Create Frequency Distribution of all words
from nltk import FreqDist

all_words_frequency = FreqDist(all_words)

print(all_words_frequency)
# <FreqDist with 39768 samples and 1583820 outcomes>


# iterating through freqdist
for word,number in all_words_frequency.items():
    print(word,":",number)

#swing : 13
# gift : 55
# telekinesis : 2
# powers : 115
# continually : 28
# endanger : 2
# ultimate : 77
# plan : 169


# print 10 most frequently occurring words
print (all_words_frequency.most_common(10))
# [(',', 77717), ('the', 76529), ('.', 65876), ('a', 38106), ('and', 35576), ('of', 34123), ('to', 31937), ("'", 30585), ('is', 25195), ('in', 21822)]

# removing stopwords

from nltk.corpus import stopwords

stopwords_english = stopwords.words('english')
print(stopwords_english)

# ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your']

all_words_without_stopwords = []
for word in all_words:
    if word not in stopwords_english:
        all_words_without_stopwords.append(word)

print(all_words_without_stopwords[:10])

# Remove Punctuation
import string

print(string.punctuation)
#!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~


# create a new list of words by removing punctuation from all_words
all_words_without_punctuation = [word for word in all_words if word not in string.punctuation]

# print the first 10 words
print(all_words_without_punctuation[:10])

# Remove both Stopwords & Punctuation

# Let's name the new list as all_words_clean
# because we clean stopwords and punctuations from the word list

all_words_clean = []
for word in all_words:
    if word not in stopwords_english and word not in string.punctuation:
        all_words_clean.append(word)

print("all words clean:",all_words_clean[:10])

# Frequency Distribution of cleaned words list

all_words_frequency = FreqDist(all_words_clean)

print(all_words_frequency)
#<FreqDist with 39586 samples and 710578 outcomes>


# print 10 most frequently occurring words
print("Most common cleaned words: ",all_words_frequency.most_common(10))
# Most common cleaned words:  [('film', 9517), ('one', 5852), ('movie', 5771), ('like', 3690), ('even', 2565), ('good', 2411)

# Create Word Feature using 2000 most frequently occurring words
print(len(all_words_frequency))  # Output: 39586

# get 2000 frequently occuring words
most_common_words = all_words_frequency.most_common(2000)
print("Most common words: ",most_common_words[:10])


# most_common_words = [word for word,frequency in most_common_words.items()]
# print("most common words: ",most_common_words)


# the most common words list's elements are in the form of tuple
# get only the first element of each tuple of the word list
word_features = [item[0] for item in most_common_words]
print(word_features[:10])

# ['film', 'one', 'movie', 'like', 'even', 'time', 'good', 'story', 'would', 'much']

# Create Feature Set
def document_features(document):
    # "set" function will remove repeated/duplicate tokens in the given list
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# get the first negative movie review file
movie_review_file = movie_reviews.fileids('neg')[0]
print(movie_review_file)

print (document_features(movie_reviews.words(movie_review_file)))
# {'contains(waste)': False, 'contains(lot)': False, 'contains(rent)': False, 'contains(black)': False, 'contains(rated)': True}

#In the beginning of this article,
# we have created the documents list which contains data of all the movie reviews.
# Its elements are tuples with word list as first item
# and review category as the second item of the tuple.

# print first tuple of the documents list
print (documents[0])
# (['plot', ':', 'two', 'teen', 'couples', 'go', ...], 'neg')

# We now loop through the documents list
# and create a feature set list using the document_features function defined above.

# convert doc to document_features(doc)
feature_set = [(document_features(doc), category) for (doc, category) in documents]
print (feature_set[0])

# Creating Train and Test Dataset
print(len(feature_set))  # Output: 2000

test_set = feature_set[:400]
train_set = feature_set[400:]

print(len(train_set))  # Output: 1600
print(len(test_set))  # Output: 400

# 2000
# 1600
# 400

# Training a Classifier
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_set)

# Testing the trained Classifier
from nltk import classify

accuracy = classify.accuracy(classifier, test_set)
print(accuracy)  # Output: 0.77

from nltk.tokenize import word_tokenize

custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = document_features(custom_review_tokens)
print(classifier.classify(custom_review_set))  # Output: neg
# Negative review correctly classified as negative

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print (prob_result) # Output: <ProbDist with 2 samples>
print (prob_result.max()) # Output: neg
print (prob_result.prob("neg")) # Output: 0.999989264571
print (prob_result.prob("pos")) # Output: 1.07354285262e-05

custom_review = "I loved the film. It was wonderful. great direction, good acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = document_features(custom_review_tokens)
print(classifier.classify(custom_review_set))  # Output: pos

# show 5 most informative features
print(classifier.show_most_informative_features(10))

# Most Informative Features
#    contains(outstanding) = True              pos : neg    =     14.7 : 1.0
#          contains(mulan) = True              pos : neg    =      7.8 : 1.0
#         contains(poorly) = True              neg : pos    =      7.7 : 1.0
#    contains(wonderfully) = True              pos : neg    =      7.5 : 1.0
#         contains(seagal) = True              neg : pos    =      6.5 : 1.0
#          contains(awful) = True              neg : pos    =      6.1 : 1.0
#         contains(wasted) = True              neg : pos    =      6.1 : 1.0
#          contains(waste) = True              neg : pos    =      5.6 : 1.0
#          contains(damon) = True              pos : neg    =      5.3 : 1.0
#          contains(flynt) = True              pos : neg    =      5.1 : 1.0

# The result shows that the word outstanding is used in positive reviews
#  14.7 times more often than it is used in negative reviews
# the word poorly is used in negative reviews 7.7 times
#  more often than it is used in positive reviews.
# Similarly, for other letters.
# These ratios are also called likelihood ratios.

# Therefore, a review has a high chance
# to be classified as positive if it contains words like outstanding and wonderfully.
# Similarly, a review has a high chance of being classified
# as negative if it contains words like poorly, awful, waste, etc.

# Bag of Words Feature
# In the above example, we used top-N words feature.
# We used 2000 most frequently occurring words as our top-N words feature.
# The top-N words feature is also a bag-of-words feature.
# But in the top-N feature, we only used the top 2000 words in the feature set.
# We combined the positive and negative reviews into a single list, randomized the list, and then separated the train and test set.
# This approach can result in the un-even distribution of positive and negative reviews across the train and test set

# Bag-of-words feature shown below

#In the bag-of-words feature as shown below:

# We will use all the useful words of each review while creating the feature set.
# We take a fixed number of positive and negative reviews for train and test set.
# This result in equal distribution of positive and negative reviews across train and test set.

# In the approach shown below, we will modify the feature extractor function.

# We form a list of unique words of each review.
# The category (pos or neg) is assigned to each bag of words.
# Then the category of any given text is calculated by matching the different bag-of-words
# & their respective category.

from nltk.corpus import movie_reviews

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append(words)

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_reviews.append(words)

# print first positive review item from the pos_reviews list
print(pos_reviews[0])

# print first negative review item from the neg_reviews list
print (neg_reviews[0])

# Feature Extraction

#We use the bag-of-words feature.
# Here, we clean the word list (i.e. remove stop words and punctuation).
# Then, we create a dictionary of cleaned words.

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

# print first positive review item from the pos_reviews list
print(pos_reviews_set[0])

# print first negative review item from the neg_reviews list
print (neg_reviews_set[0])

print(len(pos_reviews_set), len(neg_reviews_set))  # Output: (1000, 1000)

# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program
from random import shuffle

shuffle(pos_reviews_set)
shuffle(neg_reviews_set)

test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
train_set = pos_reviews_set[200:] + neg_reviews_set[200:]

print(len(test_set), len(train_set))  # Output: (400, 1600)

# Training Classifier and Calculating Accuracy

from nltk import classify
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set)
print(accuracy)  # Output: 0.7325

print(classifier.show_most_informative_features(10))

# Testing Classifier with Custom Review

from nltk.tokenize import word_tokenize

custom_review = "I hated the film. It was a disaster. Poor direction, bad acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)
print(classifier.classify(custom_review_set))  # Output: neg
# Negative review correctly classified as negative

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print(prob_result)  # Output: <ProbDist with 2 samples>
print(prob_result.max())  # Output: neg
print(prob_result.prob("neg"))  # Output: 0.776128854994
print(prob_result.prob("pos"))  # Output: 0.223871145006

custom_review = "It was a wonderful and amazing movie. I loved it. Best direction, good acting."
custom_review_tokens = word_tokenize(custom_review)
custom_review_set = bag_of_words(custom_review_tokens)

print(classifier.classify(custom_review_set))  # Output: pos
# Positive review correctly classified as positive

# probability result
prob_result = classifier.prob_classify(custom_review_set)
print(prob_result)  # Output: <ProbDist with 2 samples>
print(prob_result.max())  # Output: pos
print(prob_result.prob("neg"))  # Output: 0.0972171562901
print(prob_result.prob("pos"))  # Output: 0.90278284371
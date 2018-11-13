# https://becominghuman.ai/word-embeddings-with-gensim-68e6322afdca
# https://mubaris.com/posts/word2vec/
from gensim.models.word2vec import Word2Vec

# Download Project Gutenberg Corpus
# import nltk
# nltk.download('brown')
from nltk.corpus import brown
#sentences = brown.sents()

#example_sentence = [['this', 'is', 'a', 'sentence'], ['second', 'sentence'], ['another', 'sentence']]

#print(len(sentences))

# create word2vec model
# model = Word2Vec(sentences, min_count=3)
# print(model)
# model.save("brown_w2v.bin")
model = Word2Vec.load("brown_w2v.bin")

# number of unique words = vocab
#print(list(model.wv.vocab))

print(model.wv['cow'])

#print(model.wv.similarity("taste","sweet"))
print(list(model.wv.most_similar("name")))



from gensim.models.doc2vec import Doc2Vec,TaggedDocument

#  The input for a Doc2Vec model should be a list of TaggedDocument(['list','of','word'], [TAG_001]).
#  A good practice is using the indexes of sentences as the tags.
#  For example, to train a Doc2Vec model with two sentences (i.e. documents, paragraphs):


s1 = 'the quick fox brown fox jumps over the lazy dog'
s1_tag = '001'
s2 = 'i want to burn a zero-day'
s2_tag = '002'

docs = []
docs.append(TaggedDocument(words=s1.split(), tags=[s1_tag]))
docs.append(TaggedDocument(words=s2.split(), tags=[s2_tag]))

model = Doc2Vec(vector_size=5, window=5, min_count=1, workers=4)
model.build_vocab(docs)


# Get the vectors

print(model.docvecs[0])
print(model.docvecs[1])

# check similarity

#You need to use infer_vector to get a document vector of the new text - which does not alter the underlying model.

#Here is how you do it:

sentence_to_check = "the quick fox brown fox".split()

new_vector = model.infer_vector(sentence_to_check)
sims = model.docvecs.most_similar([new_vector],topn=2) #gives you top-most document tags and their cosine similarity
print(sims)

# [('001', 0.16089314222335815), ('002', 0.09776248782873154)]
# the score is similarity score, not distance
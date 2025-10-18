import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)
print(word_to_id)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(C)

c0 = C[word_to_id['you']]  # you的单词向量
c1 = C[word_to_id['i']]    # i的单词向量
print(cos_similarity(c0, c1))

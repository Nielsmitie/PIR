from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import pandas as pd

from project.embeddings.tag_embeddings import TagEmbedding


class GloveEmbedding(TagEmbedding):

    def __init__(self):
        self.bookmark_infos = pd.read_csv(self.path + 'bookmarks.dat',
                                          sep='\t',
                                          index_col=['id'],
                                          usecols=['id', 'title'],
                                          encoding='ISO-8859-15')
        # super(GloveEmbedding, self).__init__()
        glove_file = '../data/glove.6B/glove.6B.100d.txt'
        tmp_file = get_tmpfile('test_word2vec.txt')

        glove2word2vec(glove_file, tmp_file)

        self.model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)

        self.word_df = self._get_words_by_id()
        self.word_lists = self.word_df.values.tolist()

    def queryExp(self, query):
        expanded_query = self.model.wv.most_similar(positive=query, topn=2)

        for i in range(len(expanded_query)):
            query.append(expanded_query[i][0])
        return query

if __name__ == '__main__':
    print(GloveEmbedding().search('firefox'))

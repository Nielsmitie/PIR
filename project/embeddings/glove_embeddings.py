from gensim.models import Word2Vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import pandas as pd

from project.embeddings.tag_embeddings import TagEmbedding


class GloveEmbedding(TagEmbedding):

    def __init__(self, glove_file):
        self.bookmark_infos = pd.read_csv(self.path + 'bookmarks.dat',
                                          sep='\t',
                                          index_col=['id'],
                                          usecols=['id', 'title', 'urlPrincipal', 'url'],
                                          encoding='ISO-8859-15')
        # super(GloveEmbedding, self).__init__()

        tmp_file = get_tmpfile('test_word2vec.txt')

        glove2word2vec(glove_file, tmp_file)

        self.model = KeyedVectors.load_word2vec_format(tmp_file, binary=False)

        self.word_df = self._get_words_by_id()
        self.bookmark_infos = self.bookmark_infos.join(self.word_df)
        self.word_lists = self.word_df.values.tolist()


if __name__ == '__main__':
    glove_file = ['../data/glove.6B/glove.6B.100d.txt', '../data/glove.42B.300d.txt', '../data/glove.840B.300d.txt',
                  '../data/glove.6B/glove.6B.50d.txt', '../data/glove.6B/glove.6B.200d.txt',
                  '../data/glove.6B/glove.6B.500d.txt']
    print(GloveEmbedding(glove_file[0]).search('firefox addons extensions'.split(' ')))

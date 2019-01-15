from gensim.models import Word2Vec

import pandas as pd
import operator


class TagEmbedding:
    '''
    simple search engine. Reads in all bookmarks for each website. The generated embeddings are used to calculate
    the cosine similarities between each site and the query. The top n results are returned by self.search.

    Features to improve the performance:
    - Weighting the tags, depending on how many user tagged the site with the same tag
    - tf-idf weighting, to penalize often used tags
    - Query expansion
    - Query expansion based on users with similar behaviour
    - multiple levels of search and filter methods to improve speed

    Also some evaluation methods should be added.

    '''

    path = 'data/hackathon_02_hetrec2011-delicious-2k/'

    def __init__(self):
        self.bookmark_infos = pd.read_csv(self.path + 'bookmarks.dat',
                                          sep='\t',
                                          index_col=['id'],
                                          usecols=['id', 'title'],
                                          encoding='ISO-8859-15')

        self.word_df = self._get_words_by_id()
        self.word_lists = self.word_df.values.tolist()

        self.model = Word2Vec(window=5, workers=-1, size=100, min_count=1)
        self.model.build_vocab(self.word_lists)
        self.model.train(self.word_lists, total_examples=self.model.corpus_count, epochs=self.model.iter)

    def search(self, query, top_n=10):
        result_dict = {}
        for word_list, content in zip(self.word_lists, self.word_df.index):
            result_dict[content] = self.model.wv.n_similarity(query, word_list)

        sorted_x = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)
        # top 10 ids [19834, 98971, 2567, 9046, 13733, 27185, 53041, 68534, 89236, 95470]
        sorted_tuple = sorted_x[:top_n]
        top_n = [x[0] for x in sorted_tuple]

        return self.bookmark_infos.loc[top_n]

    def _get_words_by_id(self):
        url_tags = pd.read_csv(self.path + 'bookmark_tags.dat',
                               sep='\t',
                               index_col=['bookmarkID'],
                               usecols=['bookmarkID', 'tagID'])
        url_tags.columns = ['id']
        id_to_tag = pd.read_csv(self.path + 'tags.dat',
                                sep='\t',
                                index_col=['id'],
                                usecols=['id', 'value'],
                                encoding='ISO-8859-15')

        id_words = url_tags.join(id_to_tag, on=['id']).drop(columns=['id'])

        def test(x):
            return [i[0] for i in x.values.tolist()]

        return id_words.groupby(level=0).apply(test)

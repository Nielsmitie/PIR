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
                                          usecols=['id', 'title', 'urlPrincipal', 'url'],
                                          encoding='ISO-8859-15')

        self.word_df = self._get_words_by_id()
        self.bookmark_infos = self.bookmark_infos.join(self.word_df)
        self.word_lists = self.word_df.values.tolist()

        self.model = Word2Vec(window=5, workers=-1, size=100, min_count=1)
        self.model.build_vocab(self.word_lists)
        self.model.train(self.word_lists, total_examples=self.model.corpus_count, epochs=self.model.iter)
        self.model = self.model.wv

    def search(self, query, top_n=10):
        for word_list, content in zip(self.word_lists, self.word_df.index):
            word_list = list(filter(lambda x: x in self.model.vocab, word_list))
            if len(word_list) == 0:
                continue
            result_dict[content] = self.model.n_similarity(query, word_list)

        sorted_x = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)
        # top 10 ids [19834, 98971, 2567, 9046, 13733, 27185, 53041, 68534, 89236, 95470]
        sorted_tuple = sorted_x[:top_n]
        top_n = [x[0] for x in sorted_tuple]

        return self.bookmark_infos.loc[top_n]

    def rank(self, query):
        result_dict = {}
        query = list(filter(lambda x: x in self.model.vocab, query))
        if len(query) == 0:
            print('NO RESULTS FOUND')
            return pd.DataFrame()
        for word_list, content in zip(self.word_lists, self.word_df.index):
            word_list = list(filter(lambda x: x in self.model.vocab, word_list))
            if len(word_list) == 0:
                continue
            result_dict[content] = self.model.n_similarity(query, word_list)

        sorted_x = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)

        return pd.DataFrame(sorted_x, columns=['id', 'similarity']).set_index('id').join(self.bookmark_infos)

    def multiprocessing_rank(self, query):
        from multiprocessing import Pool

        with Pool(5) as p:
            results = p.starmap(self.worker, zip(query * len(self.word_lists), self.word_lists))

        result_dict = dict(zip(self.word_df.index, results))

        sorted_x = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)

        return pd.DataFrame(sorted_x, columns=['id', 'similarity']).set_index('id').join(self.bookmark_infos)

    def worker(self, query, word_list):
        word_list = list(filter(lambda x: x in self.model.vocab, word_list))
        if len(word_list) == 0:
            return 0
        tmp = self.model.n_similarity(query, word_list)
        return tmp

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
        tmp = id_words.groupby(level=0).apply(test)
        tmp.name = 'words_by_id'
        return tmp

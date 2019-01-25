from project.embeddings import tag_embeddings
from gensim.summarization.bm25 import get_bm25_weights
import pandas as pd

from gensim.summarization.bm25 import BM25


class QueryExpansion(tag_embeddings.TagEmbedding):


    def __init__(self):
        tag_embeddings.TagEmbedding.__init__(self)

    def queryExp(self, query):
        expanded_query = self.model.wv.most_similar(positive=query, topn=2)

        for i in range(len(expanded_query)):
            query.append(expanded_query[i][0])
        return query

    def bm25_weighting(self, query):
        bm25 = BM25(self.word_lists)

        average_idf = sum(map(lambda k: float(bm25.idf[k]), bm25.idf.keys())) / len(bm25.idf.keys())
        bm25_list = bm25.get_scores(query, average_idf)
        bm25_dataframe = pd.DataFrame(bm25_list, columns=["value"]).sort_values("value", ascending=False)
        bm25_dataframe.reset_index(inplace=True)
        bm25_datalist = bm25_dataframe.values.tolist()

        return_list = list()
        id_list = list()
        for i in range(10):
            return_list.append(bm25_datalist[i])
            id_list.append(bm25_datalist[i][0])

        return self.bookmark_infos.loc[id_list]

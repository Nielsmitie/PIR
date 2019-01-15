import pandas as pd
import numpy as np

class RecommondationEngine:

    def __init__(self):
        self.bookmark_infos = pd.read_csv(self.path + 'bookmarks.dat',
                                          sep='\t',
                                          index_col=['id'],
                                          usecols=['id', 'title'],
                                          encoding='ISO-8859-15')

        self.word_df = self._get_words_by_id()



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
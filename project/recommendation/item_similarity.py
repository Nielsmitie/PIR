import pandas as pd
import numpy as np


class RecommendationEngine:

    path = '../data/hackathon_02_hetrec2011-delicious-2k/'

    def __init__(self):
        self.bookmark_infos = pd.read_csv(self.path + 'bookmarks.dat',
                                          sep='\t',
                                          index_col=['id'],
                                          usecols=['id', 'title'],
                                          encoding='ISO-8859-15')
        # load website and tags
        self.word_df = self._get_words_by_id()
        # load user and preferences

        print('test')

    def _get_words_by_id(self):
        url_tags = pd.read_csv(self.path + 'bookmark_tags.dat',
                               sep='\t',
                               index_col=['bookmarkID'],
                               usecols=['bookmarkID', 'tagID', 'tagWeight'])
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


if __name__ == '__main__':
    RecommendationEngine()

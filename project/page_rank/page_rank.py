import pandas as pd

from project.embeddings.tag_embeddings import TagEmbedding
from project.const import file_prefix


class PageRank:

    path = file_prefix + 'data/hackathon_02_hetrec2011-delicious-2k/'

    def __init__(self):
        self.url_tags = pd.read_csv(self.path + 'bookmark_tags.dat',
                                    sep='\t',
                                    index_col=['bookmarkID'],
                                    usecols=['bookmarkID', 'tagWeight'])
        self.url_tags = self.url_tags.groupby('bookmarkID').sum()

    def page_rank(self, similarity_ranking, min_k=20):
        test = self.url_tags.join(similarity_ranking, on=['bookmarkID'])

        sorted = test.sort_values(by=['similarity', 'tagWeight'], ascending=False, na_position='last')
        min_score = 0.9
        while True:
            tmp = sorted.loc[sorted['similarity'] > min_score]
            if len(tmp.index) > min_k:
                break
            min_score -= 0.1
        sorted = tmp.sort_values(by=['tagWeight', 'similarity'], ascending=False)
        return sorted


if __name__ == '__main__':
    similarity_ranking = TagEmbedding().rank('firefox addons extensions'.split(' '))

    print(PageRank().page_rank(similarity_ranking))

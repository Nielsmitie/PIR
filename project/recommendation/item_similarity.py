import pandas as pd
import numpy as np
import turicreate as tc
import os.path


class RecommendationEngine:

    path = 'data/hackathon_02_hetrec2011-delicious-2k/'
    save_path = 'data/recommend.model'

    def __init__(self):

        if os.path.isdir(self.save_path):

            self.m2 = tc.load_model(self.save_path)
        else:
            sf = tc.SFrame.read_csv(self.path + 'user_taggedbookmarks-timestamps.dat',
                                    sep='\t',
                                    usecols=['userID', 'bookmarkID'])

            # tc item_similarity doesn't support additional information yet. So the results are
            # pretty stupid

            m = tc.item_similarity_recommender.create(sf, user_id='userID', item_id='bookmarkID', target_memory_usage=2863311530)
            nn = m.get_similar_items()
            self.m2 = tc.item_similarity_recommender.create(sf, nearest_items=nn, user_id='userID', item_id='bookmarkID', target_memory_usage=2863311530)
            self.m2.save(self.save_path)
            print('test')

    def recommend(self, item):
        return self.m2.get_similar_items(item, k=5)


if __name__ == '__main__':
    RecommendationEngine().recommend()

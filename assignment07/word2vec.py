from gensim.models import Word2Vec

import pandas as pd
import operator

url_tags = pd.read_csv('data/hetrec2011-delicious-2k/bookmark_tags.dat',
                       sep='\t',
                       index_col=['bookmarkID'],
                       usecols=['bookmarkID', 'tagID'])
url_tags.columns = ['id']
id_to_tag = pd.read_csv('data/hetrec2011-delicious-2k/tags.dat',
                        sep='\t',
                        index_col=['id'],
                        usecols=['id', 'value'],
                        encoding='ISO-8859-15')

id_words = url_tags.join(id_to_tag, on=['id']).drop(columns=['id'])

bookmark_infos = pd.read_csv('data/hetrec2011-delicious-2k/bookmarks.dat',
                             sep='\t',
                             index_col=['id'],
                             usecols=['id', 'title'],
                             encoding='ISO-8859-15')


def test(x):
    return [i[0] for i in x.values.tolist()]


word_df = id_words.groupby(level=0).apply(test)
word_lists = word_df.values.tolist()

model = Word2Vec(window=5, workers=-1, size=100, min_count=1)
model.build_vocab(word_lists)
model.train(word_lists, total_examples=model.corpus_count, epochs=model.iter)

query = ['firefox', 'addons', 'extensions']

# this solution finds just the embedding vector that is closest to the query vector. Both are used with embeddings
result_dict = {}
for word_list, content in zip(word_lists, word_df.index):
    result_dict[content] = model.wv.n_similarity(query, word_list)

# top ten results
k = 10

sorted_x = sorted(result_dict.items(), key=operator.itemgetter(1), reverse=True)

sorted_tuple = sorted_x[:k]
top_10 = [x[0] for x in sorted_tuple]

# top 10 ids [19834, 98971, 2567, 9046, 13733, 27185, 53041, 68534, 89236, 95470]
print('Top 10 Url titles:')
print(bookmark_infos.loc[top_10])

'''
Final result
Top 10 Url titles:
                                                    title
id                                                       
19834       Programador de agenda de reuniones para Gmail
105823         30+ Must-Have Updated Firefox 3 Extensions
14422                 Top 10 Must-Have Browser Extensions
22359                 Top 10 Must-Have Browser Extensions
16933                     DownThemAll! :: Firefox Add-ons
60678                      LinkChecker :: Firefox Add-ons
37924             Nightly Tester Tools :: Firefox Add-ons
82547   Add-on compatibility for Firefox 4 â time to...
37922                     Adblock Plus :: Firefox Add-ons
96                             Plugins :: Firefox Add-ons
'''

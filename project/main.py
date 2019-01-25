from project.embeddings import tag_embeddings
from project.embeddings import glove_embeddings
from project.page_rank import page_rank
from project.query_expansion import expansion


def print_results(df):
    for index, row in df.iterrows():
        print('{}\n{}\n{}\n{}'.format(row['title'], row['url'], row['urlPrincipal'], row['words_by_id']))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    print('Loading Search Engine')

    searcher = glove_embeddings.GloveEmbedding('data/glove.6B/glove.6B.100d.txt')
    ranker = page_rank.PageRank()

    print('-'*20)

    print('Welcome to query City:')
    query = 'start'
    while query != 'exit':
        query = input('pose your query:').lower()
        query = query.split(' ')

        print('Top 10 Website titles:')

        top_results = searcher.rank(query)
        if top_results.empty:
            continue

        top_10 = ranker.page_rank(top_results, top_k=10)

        print_results(top_10)

    expanded_query = expansion.QueryExpansion().queryExp(query)

    print(expansion.QueryExpansion().bm25_weighting(query))
    print('End of results')
    print('-'*20)

    print('')
    print('Ending session')
    print('-'*20)


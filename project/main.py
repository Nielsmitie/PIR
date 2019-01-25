from project.embeddings import glove_embeddings
from project.page_rank import page_rank


def print_results(df, i=0):
    for index, row in df.iterrows():
        print('{}: {}\n{}\n{}\n{}\n'.format(i, row['title'], row['url'], row['urlPrincipal'], row['words_by_id']))
        i += 1


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    print('Loading Search Engine')

    searcher = glove_embeddings.GloveEmbedding('data/glove.6B/glove.6B.100d.txt')
    ranker = page_rank.PageRank()

    print('-'*20)

    print('Welcome')
    print('')
    query = ['start']
    while True:
        query = input('Pose your query: ').lower()
        if 'exit' in query:
            break
        query = query.split(' ')
        query = searcher.queryExp(query)

        print('Top 10 Websites:')

        top_results = searcher.rank(query)
        if top_results.empty:
            continue

        top_rank = ranker.page_rank(top_results, min_k=50)

        num_of_results = 10
        start = 0
        while True:
            try:
                print_results(top_rank.iloc[start:num_of_results], i=start+1)
            except IndexError:
                print('End of results]')
                break
            control = input('Get more results with [+]\nDiscontinue Query with [-]\n: ')

            if control == '+':
                num_of_results += 10
                start += 10
            else:
                break

        print('End of results')
        print('-'*20)
        print()
    print('')
    print('Ending session')
    print('-'*20)


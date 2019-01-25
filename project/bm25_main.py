from project.query_expansion import expansion

def print_results(df):
    for index, row in df.iterrows():
        print('{}\n{}\n{}\n{}'.format(row['title'], row['url'], row['urlPrincipal'], row['words_by_id']))

if __name__ == '__main__':
    query = input('pose your query:').lower()
    query = query.split(' ')

    bm25_weight = expansion.QueryExpansion().bm25_weighting(query)

    print_results(bm25_weight)
    print(bm25_weight)
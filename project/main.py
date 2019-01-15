from project import search_engine

if __name__ == '__main__':
    query = input('pose your query:').lower()
    query = query.split(' ')

    print('Top 10 Url titles:')
    print(search_engine.SearchEngine().search(query))


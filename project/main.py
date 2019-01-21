from project.embeddings import tag_embeddings
from project.query_expansion import expansion

if __name__ == '__main__':
    query = input('pose your query:').lower()
    query = query.split(' ')

    print(expansion.QueryExpansion().queryExp(query))

    print('Top 10 Url titles:')
    print(tag_embeddings.TagEmbedding().search(expansion.QueryExpansion().queryExp(query)))


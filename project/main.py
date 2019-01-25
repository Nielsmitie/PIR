from project.embeddings import tag_embeddings
from project.query_expansion import expansion

if __name__ == '__main__':
    query = input('pose your query:').lower()
    query = query.split(' ')

    expanded_query = expansion.QueryExpansion().queryExp(query)

    print(expansion.QueryExpansion().bm25_weighting(query))

    print('Top 10 Url titles:')
    print(tag_embeddings.TagEmbedding().search(expansion.QueryExpansion().queryExp(query)))


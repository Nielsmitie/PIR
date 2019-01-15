from project.embeddings import tag_embeddings

if __name__ == '__main__':
    query = input('pose your query:').lower()
    query = query.split(' ')

    print('Top 10 Url titles:')
    print(tag_embeddings.TagEmbedding().search(query))


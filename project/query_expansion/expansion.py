from project.embeddings import tag_embeddings

class QueryExpansion(tag_embeddings.TagEmbedding):


    def __init__(self):
        tag_embeddings.TagEmbedding.__init__(self)

    def queryExp(self, query):
        expanded_query = self.model.wv.most_similar(positive=query, topn=2)

        for i in range(len(expanded_query)):
            query.append(expanded_query[i][0])
        return query
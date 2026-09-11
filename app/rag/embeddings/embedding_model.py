from sentence_transformers import SentenceTransformer

class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts: list[str]):
        """
        Convert multiple text chunks into embeddings.
        """

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        return embeddings

    def embed_query(self, query: str):
        """
        Convert a user question into an embedding.
        """

        embedding = self.model.encode(query)

        return embedding
from app.rag.embeddings.embedding_model import EmbeddingModel


chunks = [
    "Dubai Marina is a waterfront community in Dubai.",
    "Dubai Building Code defines construction requirements."
]


embedding_model = EmbeddingModel()


embeddings = embedding_model.embed_documents(
    chunks
)


print("\nTOTAL EMBEDDINGS:")
print(len(embeddings))


print("\nEMBEDDING DIMENSION:")
print(len(embeddings[0]))


print("\nFIRST EMBEDDING:")
print(embeddings[0][:10])
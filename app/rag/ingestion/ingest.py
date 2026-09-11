from app.rag.ingestion.source_detector import SourceDetector
from app.rag.ingestion.url_fetcher import URLFetcher
from app.rag.ingestion.content_extractor import ContentExtractor
from app.rag.ingestion.text_cleaner import TextCleaner
from app.rag.ingestion.chunker import TextChunker

from app.rag.embeddings.embedding_model import EmbeddingModel


class DocumentIngestionPipeline:

    def __init__(self):

        self.source_detector = SourceDetector()
        self.url_fetcher = URLFetcher()
        self.content_extractor = ContentExtractor()
        self.text_cleaner = TextCleaner()
        self.chunker = TextChunker()
        self.embedding_model = EmbeddingModel()

    # =========================================================
    # PROCESS DOCUMENT URL
    # =========================================================

    def process_url(
        self,
        document_id: str,
        document_title: str,
        document_url: str,
        publisher: str = None
    ):

        print("\n================================")
        print("DOCUMENT INGESTION")
        print("================================")

        print(f"Document ID: {document_id}")
        print(f"Document Title: {document_title}")
        print(f"Document URL: {document_url}")

        # -----------------------------------------------------
        # 1. SOURCE DETECTION
        # -----------------------------------------------------

        source_type = self.source_detector.detect(
            document_url
        )

        print(f"Source Type: {source_type}")

        if source_type == "unknown":

            print("Unknown source type. Skipping.")

            return []

        # -----------------------------------------------------
        # 2. FETCH
        # -----------------------------------------------------

        fetched = self.url_fetcher.fetch(
            document_url
        )

        # -----------------------------------------------------
        # 3. EXTRACT
        # -----------------------------------------------------

        raw_content = self.content_extractor.extract(
            content=fetched["content"],
            content_type=fetched["content_type"],
            source_type=source_type
        )

        if not raw_content or not raw_content.strip():

            print("No content extracted. Skipping.")

            return []

        # -----------------------------------------------------
        # 4. CLEAN
        # -----------------------------------------------------

        cleaned_content = self.text_cleaner.clean(
            raw_content
        )

        if not cleaned_content or not cleaned_content.strip():

            print("No content after cleaning. Skipping.")

            return []

        # -----------------------------------------------------
        # 5. CHUNK
        # -----------------------------------------------------

        chunks = self.chunker.chunk(
            cleaned_content
        )

        print("\n================================")
        print("CHUNKING RESULT")
        print("================================")

        print(f"Total chunks: {len(chunks)}")

        if not chunks:

            print("No chunks created.")

            return []

        # -----------------------------------------------------
        # 6. EMBEDDING
        # -----------------------------------------------------

        print("\n================================")
        print("GENERATING EMBEDDINGS")
        print("================================")

        embeddings = self.embedding_model.embed_documents(
            chunks
        )

        if len(embeddings) != len(chunks):

            raise RuntimeError(
                "Embedding count does not match "
                "chunk count."
            )

        embedding_dimension = len(
            embeddings[0]
        )

        print(
            f"Generated embeddings: "
            f"{len(embeddings)}"
        )

        print(
            f"Embedding dimension: "
            f"{embedding_dimension}"
        )

        if embedding_dimension != 384:

            raise RuntimeError(
                f"Expected 384 dimensions, "
                f"got {embedding_dimension}."
            )

        # -----------------------------------------------------
        # 7. BUILD RESULTS
        # -----------------------------------------------------

        results = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings),
            start=1
        ):

            results.append({

                "text": chunk,

                "embedding":
                    embedding.tolist(),

                "metadata": {

                    "document_id":
                        document_id,

                    "document_title":
                        document_title,

                    "document_type":
                        source_type,

                    "document_url":
                        document_url,

                    "final_url":
                        fetched["final_url"],

                    "publisher":
                        publisher,

                    "chunk_id":
                        index
                }
            })

        # -----------------------------------------------------
        # 8. COMPLETE
        # -----------------------------------------------------

        print("\n================================")
        print("INGESTION COMPLETE")
        print("================================")

        print(
            f"Document: {document_title}"
        )

        print(
            f"Chunks: {len(results)}"
        )

        print(
            f"Embedding dimension: "
            f"{embedding_dimension}"
        )

        print(
            "Status: Ready for Supabase pgvector"
        )

        return results


# =============================================================
# SIMPLE URL FUNCTION
# =============================================================

def ingest_url(
    document_id: str,
    document_title: str,
    document_url: str,
    publisher: str = None
):

    pipeline = DocumentIngestionPipeline()

    return pipeline.process_url(
        document_id=document_id,
        document_title=document_title,
        document_url=document_url,
        publisher=publisher
    )
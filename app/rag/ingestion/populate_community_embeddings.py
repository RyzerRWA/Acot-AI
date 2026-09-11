import sys
from pathlib import Path

# Make sure the project root is available when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


print("\n================================")
print("COMMUNITY EMBEDDING INGESTION")
print("================================")


from app.database.supabase_client import supabase
from app.rag.embeddings.embedding_model import EmbeddingModel


class CommunityEmbeddingIngestor:

    def __init__(self):
        print("\nLoading embedding model...")
        self.embedding_model = EmbeddingModel()
        print("Embedding model loaded.")

    def get_records_without_embeddings(self):
        print("\nFetching communities from Supabase...")

        response = (
            supabase
            .table("communities")
            .select("id, name, knowledge_text")
            .is_("embedding", "null")
            .execute()
        )

        records = response.data or []

        return records

    def run(self, batch_size=50):

        records = self.get_records_without_embeddings()

        print(
            f"\nRecords without embeddings: {len(records)}"
        )

        if not records:
            print("\nAll community records already have embeddings.")
            return

        total = len(records)

        for start in range(0, total, batch_size):

            batch = records[start:start + batch_size]

            print("\n--------------------------------")
            print(
                f"Processing records "
                f"{start + 1} - {start + len(batch)} "
                f"of {total}"
            )
            print("--------------------------------")

            # Get knowledge text
            texts = []

            valid_records = []

            for record in batch:

                knowledge_text = record.get("knowledge_text")

                if not knowledge_text:
                    print(
                        f"Skipping {record['id']} - "
                        f"no knowledge_text"
                    )
                    continue

                texts.append(knowledge_text)
                valid_records.append(record)

            if not valid_records:
                continue

            # Generate embeddings
            print("Generating embeddings...")

            embeddings = (
                self.embedding_model.embed_documents(texts)
            )

            print(
                f"Generated {len(embeddings)} embeddings."
            )

            # Store embeddings
            print("Storing embeddings in Supabase...")

            for record, embedding in zip(
                valid_records,
                embeddings
            ):

                vector = embedding.tolist()

                (
                    supabase
                    .table("communities")
                    .update({
                        "embedding": vector
                    })
                    .eq(
                        "id",
                        record["id"]
                    )
                    .execute()
                )

                print(
                    f"Stored: "
                    f"{record['id']} - "
                    f"{record['name']}"
                )

        print("\n================================")
        print("COMMUNITY EMBEDDING INGESTION")
        print("COMPLETED")
        print("================================")


if __name__ == "__main__":

    try:

        ingestor = CommunityEmbeddingIngestor()

        ingestor.run(
            batch_size=50
        )

    except Exception as error:

        print("\n================================")
        print("ERROR")
        print("================================")

        print(
            f"{type(error).__name__}: {error}"
        )

        raise
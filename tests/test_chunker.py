from app.rag.loaders.url_loader import load_url_content
from app.rag.processing.cleaner import clean_text
from app.rag.processing.chunker import chunk_text


url = (
    "https://www.dm.gov.ae/"
    "municipality-business/"
    "planning-and-construction/"
    "dubai-building-code-2/"
)


# Step 1: Load URL content
raw_content = load_url_content(url)

# Step 2: Clean content
cleaned_content = clean_text(raw_content)

# Step 3: Chunk content
chunks = chunk_text(cleaned_content)


print("\nTOTAL CHUNKS:")
print(len(chunks))


print("\nFIRST CHUNK:\n")
print(chunks[0])


print("\nSECOND CHUNK:\n")
print(chunks[1])
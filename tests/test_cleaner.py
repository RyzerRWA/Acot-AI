from app.rag.loaders.url_loader import load_url_content
from app.rag.processing.cleaner import clean_text


url = (
    "https://www.dm.gov.ae/"
    "municipality-business/"
    "planning-and-construction/"
    "dubai-building-code-2/"
)


# Step 1: Extract content
raw_content = load_url_content(url)

print("\nRAW CONTENT LENGTH:")
print(len(raw_content))


# Step 2: Clean content
cleaned_content = clean_text(raw_content)

print("\nCLEANED CONTENT LENGTH:")
print(len(cleaned_content))


print("\nCLEANED CONTENT PREVIEW:\n")

print(cleaned_content[:2000])
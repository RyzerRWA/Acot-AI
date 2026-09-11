from app.database.supabase_client import supabase


response = (
    supabase
    .table("communities")
    .select("id, name, city, knowledge_text, source")
    .limit(3)
    .execute()
)

print("Rows returned:", len(response.data))

for row in response.data:
    print("\n-----------------------------")
    print("ID:", row["id"])
    print("Name:", row["name"])
    print("City:", row["city"])
    print("Source:", row["source"])
    print("Knowledge:")
    print(row["knowledge_text"])
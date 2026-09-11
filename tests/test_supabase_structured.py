import os
from dotenv import load_dotenv

from app.retrieval.supabase.structured_retriever import (
    SupabaseStructuredRetriever
)


load_dotenv()


print("\n================================")
print("SUPABASE STRUCTURED RETRIEVER TEST")
print("================================")


print("\nInitializing Supabase structured retriever...")

retriever = SupabaseStructuredRetriever()

print("Supabase structured retriever ready.")


community = "Dubai Marina"

print("\n================================")
print("SEARCHING COMMUNITY")
print("================================")

print("Community:", community)


results = retriever.retrieve(
    community_name=community
)


print("\n================================")
print("COMMUNITY RESULTS")
print("================================")

print(
    "Communities found:",
    len(results["community"])
)


for record in results["community"]:
    print("\nCommunity:")
    print("ID:", record.get("id"))
    print("Name:", record.get("name"))
    print("City:", record.get("city"))
    print("Source:", record.get("source"))


print("\n================================")
print("PROJECT RESULTS")
print("================================")

print(
    "Projects found:",
    len(results["projects"])
)


for i, project in enumerate(results["projects"], 1):

    print(f"\nProject {i}")
    print("----------------------------")

    print("ID:", project.get("id"))
    print("Name:", project.get("name"))
    print("City:", project.get("city"))
    print("Community:", project.get("community"))
    print("Sub-community:", project.get("sub_community"))
    print("Developer:", project.get("developer_name"))
    print("Price:", project.get("price"))
    print("Bedrooms:",
          project.get("bedroom_min"),
          "-",
          project.get("bedroom_max"))
    print("Property Types:",
          project.get("property_types"))
    print("Handover:", project.get("handover_time"))
    print("Property Finder:",
          project.get("property_finder_url"))
    print("Source:", project.get("source"))


print("\n================================")
print("SUB-COMMUNITY RESULTS")
print("================================")

print(
    "Sub-communities found:",
    len(results["sub_communities"])
)


for sub in results["sub_communities"]:

    print("\nSub-community:")
    print("ID:", sub.get("id"))
    print("Name:", sub.get("name"))
    print("Community:", sub.get("community"))
    print("City:", sub.get("city"))


print("\n================================")
print("TEST COMPLETED")
print("================================")
from app.retrieval.supabase.project_vector_retriever import (
    SupabaseProjectVectorRetriever
)


def main():

    print("\n================================")
    print("PROJECT VECTOR RETRIEVER TEST")
    print("================================")

    retriever = SupabaseProjectVectorRetriever(
        match_threshold=0.0,
        match_count=5
    )

    question = (
        "Tell me about luxury apartments "
        "in Dubai Marina"
    )

    results = retriever.search(
        question=question
    )

    print("\n================================")
    print("PROJECT VECTOR RESULTS")
    print("================================")

    if not results:

        print("No projects found.")

        return

    for index, project in enumerate(
        results,
        start=1
    ):

        print(
            f"\nProject {index}"
        )

        print("----------------------------")

        print(
            f"ID: {project.get('id')}"
        )

        print(
            f"Name: {project.get('name')}"
        )

        print(
            f"Developer: "
            f"{project.get('developer_name')}"
        )

        print(
            f"City: {project.get('city')}"
        )

        print(
            f"Community: "
            f"{project.get('community')}"
        )

        print(
            f"Price: "
            f"{project.get('price')}"
        )

        print(
            f"Bedrooms: "
            f"{project.get('bedroom_min')} - "
            f"{project.get('bedroom_max')}"
        )

        print(
            f"Property Types: "
            f"{project.get('property_types')}"
        )

        print(
            f"Similarity: "
            f"{project.get('similarity')}"
        )

        print(
            f"Property Finder: "
            f"{project.get('property_finder_url')}"
        )


if __name__ == "__main__":
    main()
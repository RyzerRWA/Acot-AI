from app.retrieval.hybrid.router import HybridRouter


router = HybridRouter()


questions = [

    "Tell me about Dubai Marina",

    "Show me properties in Downtown Dubai",

    "What are the Dubai Building Code safety requirements?",

    "Is Dubai Marina a good investment based on market reports?"

]


for question in questions:

    result = router.route(question)

    print("\n-------------------------")

    print("QUESTION:")
    print(question)

    print("\nROUTE:")
    print(result["route"])

    print("\nSTRUCTURED DATA:")
    print(result["structured"])

    print("\nDOCUMENT RAG:")
    print(result["documents"])
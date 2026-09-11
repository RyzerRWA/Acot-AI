from app.retrieval.hybrid.router import HybridRouter


router = HybridRouter()


questions = [
    "show me the projects in Dubai Marina",
    "show me properties in Dubai Marina",
    "what are the developers in Dubai Marina",
    "what are Dubai Building Code requirements",
    "is Dubai Marina a good investment?",
    "tell me about Dubai Marina"
]


print("\n================================")
print("ACOT ROUTER TEST")
print("================================")


for question in questions:

    result = router.route(question)

    print("\nQuestion:")
    print(question)

    print("Route:")
    print(result["route"])

    print("Question type:")
    print(result["question_type"])

    print("Structured:")
    print(result["structured"])

    print("Documents:")
    print(result["documents"])

    print("Hybrid:")
    print(result["hybrid"])


print("\n================================")
print("TEST COMPLETED")
print("================================")
from app.llm.client import GeminiClient


llm = GeminiClient()


response = llm.generate(
    "Explain Dubai real estate in one sentence."
)


print("\nGEMINI RESPONSE:\n")

print(response)
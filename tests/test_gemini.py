from app.llm.client import AICreditsClient


llm = AICreditsClient()


response = llm.generate(
    "Explain Dubai real estate in one sentence."
)


print(f"\n{llm.model} RESPONSE:\n")

print(response)
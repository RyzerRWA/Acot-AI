import asyncio

from app.llm.client import AICreditsClient


llm = AICreditsClient()


response = asyncio.run(llm.generate(
    "Explain Dubai real estate in one sentence."
))


print(f"\n{llm.model} RESPONSE:\n")

print(response)
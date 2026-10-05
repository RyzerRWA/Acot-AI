import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env.chat")

API_KEY = os.getenv("SUMMARY_AICREDITS_API_KEY")
BASE_URL = os.getenv("SUMMARY_AICREDITS_BASE_URL")
MODEL = os.getenv("SUMMARY_AICREDITS_MODEL", "z-ai/glm-5.3")

if not API_KEY or not BASE_URL:
    raise RuntimeError("Summary AI Credits configuration is missing.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_account_summary(intelligence: dict) -> str:
    prompt = f"""
You are ACOT's account-level real-estate behavior analyst.

Analyze ONLY the supplied account intelligence derived from USER QUESTIONS.

Write the summary directly about the user.

IMPORTANT:
- Start the summary with "The user..." or naturally describe the user's interests.
- Do NOT start with "This account..."
- Do NOT start with "This user shows..."
- Do NOT mention question counts.
- Do NOT mention conversation counts.
- Do NOT mention raw statistics or numbers.
- Focus on what the user has a strong interest in.
- Describe clear preferences and recurring research behavior.
- Use natural phrases such as:
  "The user has a strong interest in..."
  "The user has a clear preference for..."
  "The user frequently explores..."
  "The user appears particularly interested in..."
  "The user often compares..."
- Do not analyze assistant answers.
- Do not invent facts.
- Do not overstate conclusions.

Focus on:
locations, bedroom preferences, investment/rental yield,
prices, handover dates, amenities, comparisons, and recent interests.

FORMAT:

Summary:
Write one natural paragraph describing the user's main
real-estate interests and research behavior.
AcotScore:
give a single number between 0 and 10 that represents the user's overall engagement and interest level in real estate topics.

Key Insights:
Write 4-6 concise points describing the strongest observed interests.

ACCOUNT INTELLIGENCE:
{intelligence}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
        reasoning_effort="low",
    )

    content = response.choices[0].message.content

    if content:
        return content.strip()

    raise RuntimeError("GLM returned no final content.")
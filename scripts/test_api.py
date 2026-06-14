from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv(override=True)
print("ENV file location:", os.path.abspath(".env"))
print("Key loaded:", os.getenv("OPENAI_API_KEY")[:15] + "...")
api_key = os.getenv("OPENAI_API_KEY")
print("Key loaded:", api_key[:10] + "...")  # prints first 10 chars to verify

client = OpenAI(api_key=api_key)

# simplest possible API call, no vision, no image
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "say hello"}],
    max_tokens=10
)

print("Response:", response.choices[0].message.content)
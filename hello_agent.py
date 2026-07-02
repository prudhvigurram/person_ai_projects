import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # pulls ANTHROPIC_API_KEY from .env into env vars

client = Anthropic()  # SDK auto-reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    messages=[
        {"role": "user", "content": "Say hello and confirm you're working. One sentence."}
    ]
)
print(response.content[0].text)
print(f"\nTokens used → input: {response.usage.input_tokens}, output: {response.usage.output_tokens}")

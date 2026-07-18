import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")

print(f"API Key: {api_key[:10]}...")
print(f"Base URL: {base_url}")

client = openai.OpenAI(api_key=api_key, base_url=base_url)

try:
    models = client.models.list()
    print("\nAvailable Models:")
    for m in models.data:
        print(f"  - {m.id}")
except Exception as e:
    print(f"Error listing models: {e}")

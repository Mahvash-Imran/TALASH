import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL")

client = openai.OpenAI(api_key=api_key, base_url=base_url)

models_to_test = ["groq/compound", "qwen/qwen3-32b", "meta-llama/llama-4-scout-17b-16e-instruct", "groq/compound-mini"]

for model in models_to_test:
    print(f"\nTesting model: {model}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello! Reply in 3 words."}],
            max_tokens=10
        )
        print(f"Success! Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed: {e}")

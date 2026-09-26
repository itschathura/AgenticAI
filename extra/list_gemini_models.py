from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

for model in client.models.list():
    if "gemini" in model.name.lower():
        print(model.name)
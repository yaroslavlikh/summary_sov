from google import genai
from config import get_token_gemini
from llm.prompt import prompt_for_llm

API_key = get_token_gemini()


def send_promte(promte):
    client = genai.Client(api_key=API_key)
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt_for_llm + promte
    )
    print(response.text)
    return response.text




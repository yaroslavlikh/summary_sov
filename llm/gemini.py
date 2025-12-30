from google import genai
from config import get_token_gemini
from llm.prompt import prompt_for_llm

API_key = get_token_gemini()

def send_prompt(prompt):
    client = genai.Client(api_key=API_key)
    new = ""
    try:
        flag = True
        response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt_for_llm + prompt
        )
    except Exception as e:
        if flag:
            flag = False
            new = "Новая версия модели недоступна, перешел на старую"
        response = client.models.generate_content(
        model="gemini-flash-latest", contents=prompt_for_llm + prompt
        )
    print(response.text)
    if new: return f'{new}\n{response.text}'
    else: return response.text




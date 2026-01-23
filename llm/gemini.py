from google import genai
from config import get_token_gemini
from llm.prompt import prompt_for_llm

API_key = get_token_gemini()

def send_prompt(prompt):
    if not API_key:
        print("Ошибка: GEMINI_API_KEY не установлен")
        return None
    
    client = genai.Client(api_key=API_key)
    full_prompt = prompt_for_llm + prompt
    
    # Пытаемся использовать новую модель
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest", 
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        print(f"Ошибка при использовании gemini-flash-latest: {e}")
        print("Переход на gemini-3-flash-preview...")
        
        # Пытаемся использовать старую модель
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview", 
                contents=full_prompt
            )
            return f"Новая версия модели недоступна, перешел на старую\n{response.text}"
        except Exception as e2:
            print(f"Ошибка при использовании gemini-3-flash-preview: {e2}")
            return None
from google import genai
from config import get_token_gemini


PROMTE = """Проанализируй следующий текст и составь краткое содержание в формате списка с вложенными пунктами. Используй заголовок '#summary', затем перечисли ключевые темы обсуждения, каждую с новой строки, начиная с дефиса. Если в теме есть подпункты — оформи их с отступом в 2 пробела и также начни с дефиса. Сохраняй нейтральный и лаконичный тон, фокусируйся на сути обсуждений и действиях участников.
Текст для анализа:"""
API_key = get_token_gemini()
print(API_key)


def send_promte(promte):
    client = genai.Client(api_key=API_key)
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=PROMTE+promte
    )
    print(response.text)
    return response.text




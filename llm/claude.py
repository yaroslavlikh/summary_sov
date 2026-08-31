from anthropic import Anthropic

from config import get_anthropic_api_key
from llm.prompt import prompt_for_llm

API_key = get_anthropic_api_key()

PRIMARY_MODEL = "claude-sonnet-5"
FALLBACK_MODEL = "claude-haiku-4-5-20251001"


def _ask(client, model, full_prompt):
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": full_prompt}],
    )
    return response.content[0].text


def send_prompt(prompt, max_lines=18):
    if not API_key:
        print("Ошибка: ANTHROPIC_API_KEY не установлен")
        return None

    client = Anthropic(api_key=API_key)
    full_prompt = prompt_for_llm.format(max_lines=max_lines) + prompt

    try:
        return _ask(client, PRIMARY_MODEL, full_prompt)
    except Exception as e:
        print(f"Ошибка при использовании {PRIMARY_MODEL}: {e}")
        print(f"Переход на {FALLBACK_MODEL}...")

        try:
            return _ask(client, FALLBACK_MODEL, full_prompt)
        except Exception as e2:
            print(f"Ошибка при использовании {FALLBACK_MODEL}: {e2}")
            return None

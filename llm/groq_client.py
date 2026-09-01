from groq import Groq

from config import get_groq_api_key
from llm.prompt import prompt_for_llm

API_key = get_groq_api_key()

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"


def _ask(client, model, full_prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def _complete(full_prompt, temperature):
    if not API_key:
        print("Ошибка: GROQ_API_KEY не установлен")
        return None

    client = Groq(api_key=API_key)

    try:
        return _ask(client, PRIMARY_MODEL, full_prompt, temperature)
    except Exception as e:
        print(f"Ошибка при использовании {PRIMARY_MODEL}: {e}")
        print(f"Переход на {FALLBACK_MODEL}...")

        try:
            return _ask(client, FALLBACK_MODEL, full_prompt, temperature)
        except Exception as e2:
            print(f"Ошибка при использовании {FALLBACK_MODEL}: {e2}")
            return None


def send_prompt(prompt, max_lines=18):
    return _complete(prompt_for_llm.format(max_lines=max_lines) + prompt, temperature=0.9)


def answer_question(full_prompt):
    # Factual Q&A needs consistent, format-compliant answers, not creative
    # variation -- lower temperature than the summary path.
    return _complete(full_prompt, temperature=0.3)

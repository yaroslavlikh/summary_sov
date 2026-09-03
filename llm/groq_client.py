from groq import Groq

from config import get_groq_api_key
from llm.prompt import prompt_for_llm, prompt_for_query_rewrite, prompt_for_rerank

API_key = get_groq_api_key()

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

# Bulk history analysis (context_learning.py) can burn hundreds of requests
# in one run. Groq tracks daily quota separately per model, so giving it its
# own dedicated model here means it never eats into PRIMARY_MODEL/FALLBACK_MODEL's
# quota that /summary and /ask depend on.
CONTEXT_LEARNING_MODEL = "qwen/qwen3.8-27b"


def _ask(client, model, full_prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def _complete(full_prompt, temperature, models):
    if not API_key:
        print("Ошибка: GROQ_API_KEY не установлен")
        return None

    client = Groq(api_key=API_key)

    last_error = None
    for model in models:
        try:
            return _ask(client, model, full_prompt, temperature)
        except Exception as e:
            print(f"Ошибка при использовании {model}: {e}")
            last_error = e

    print(f"Все модели недоступны: {last_error}")
    return None


def send_prompt(prompt, max_lines=18, group_context=""):
    full_prompt = prompt_for_llm.format(max_lines=max_lines, group_context=group_context) + prompt
    return _complete(full_prompt, temperature=0.9, models=[PRIMARY_MODEL, FALLBACK_MODEL])


def answer_question(full_prompt):
    # Factual Q&A needs consistent, format-compliant answers, not creative
    # variation -- lower temperature than the summary path.
    return _complete(full_prompt, temperature=0.3, models=[PRIMARY_MODEL, FALLBACK_MODEL])


def answer_context_question(full_prompt):
    return _complete(full_prompt, temperature=0.5, models=[CONTEXT_LEARNING_MODEL])


# Both of these are small, single-purpose calls on the retrieval hot path
# (not the final answer) -- FALLBACK_MODEL alone (gpt-oss-20b, ~900+ tok/s on
# Groq) keeps them cheap and fast rather than pulling in PRIMARY_MODEL.

def rewrite_query(question, context_lines):
    context = "\n".join(context_lines)
    full_prompt = prompt_for_query_rewrite.format(context=context, question=question)
    result = _complete(full_prompt, temperature=0.2, models=[FALLBACK_MODEL])
    return result.strip() if result else None


def rerank_candidates(question, numbered_messages):
    full_prompt = prompt_for_rerank.format(question=question, messages=numbered_messages)
    return _complete(full_prompt, temperature=0.0, models=[FALLBACK_MODEL])

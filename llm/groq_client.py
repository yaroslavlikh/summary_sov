import json
import re

from groq import Groq
from langfuse import get_client, observe

from config import get_groq_api_key
from llm.prompt import (
    prompt_for_context_extraction,
    prompt_for_image_caption,
    prompt_for_llm,
    prompt_for_query_rewrite,
    prompt_for_rerank,
)

API_key = get_groq_api_key()
# get_client() reads LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST from
# the environment. If they're unset it silently no-ops instead of erroring, so
# tracing is opt-in and never breaks local runs without Langfuse configured.
langfuse = get_client()

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

# Bulk history analysis (context_learning.py) can burn hundreds of requests
# in one run. Groq tracks daily quota separately per model, so giving it its
# own dedicated model here means it never eats into PRIMARY_MODEL/FALLBACK_MODEL's
# quota that /summary and /ask depend on.
CONTEXT_LEARNING_MODEL = "qwen/qwen3.8-27b"

# gpt-oss models are text-only; these two qwen variants are the only
# vision-capable models on Groq. Using the OTHER one from CONTEXT_LEARNING_MODEL
# keeps photo/sticker captioning (which can happen far more often, on every
# image message) from competing with context-learning's quota.
VISION_MODEL = "qwen/qwen3.6-27b"


@observe(as_type="generation", name="groq-completion", capture_input=False, capture_output=False)
def _ask(client, model, full_prompt, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    usage = response.usage
    langfuse.update_current_generation(
        model=model,
        input=full_prompt,
        output=content,
        model_parameters={"temperature": temperature},
        usage_details={
            "input": usage.prompt_tokens,
            "output": usage.completion_tokens,
            "total": usage.total_tokens,
        } if usage else None,
    )
    return content


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

def rewrite_query(question, context_lines, asker_name="неизвестный"):
    context = "\n".join(context_lines)
    full_prompt = prompt_for_query_rewrite.format(context=context, question=question, asker_name=asker_name)
    result = _complete(full_prompt, temperature=0.2, models=[FALLBACK_MODEL])
    return result.strip() if result else None


def rerank_candidates(question, numbered_messages):
    full_prompt = prompt_for_rerank.format(question=question, messages=numbered_messages)
    return _complete(full_prompt, temperature=0.0, models=[FALLBACK_MODEL])


CONTEXT_EXTRACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_portrait",
            "description": (
                "Добавить факт/черту о конкретном человеке — то, что он сказал "
                "о себе, или что о нём сказали другие."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person": {"type": "string", "description": "Имя/ник человека"},
                    "addition": {"type": "string", "description": "Что добавить, коротко"},
                    "source": {"type": "string", "enum": ["self", "external"]},
                },
                "required": ["person", "addition", "source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_chat_lore",
            "description": "Добавить общий факт/шутку/повторяющуюся тему чата, не привязанную к одному человеку.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_moment",
            "description": "Записать яркий момент/мнение/шутку из переписки.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
]

# Fire-and-forget writes, not lookups -- the model never needs real data back,
# so every tool result is just a flat acknowledgement. Capped iterations
# guard against a runaway back-and-forth if the model keeps calling tools.
_MAX_TOOL_ITERATIONS = 6


@observe(name="groq-context-extraction")
def extract_context_updates(messages_text, execute_fn):
    """Runs a tool-calling loop over messages_text on FALLBACK_MODEL.
    execute_fn(tool_name, arguments_dict) is called for each tool call the
    model makes; exceptions from it are fed back to the model as the tool
    result so it can see the write failed, rather than silently vanishing."""
    if not API_key:
        print("Ошибка: GROQ_API_KEY не установлен")
        return

    client = Groq(api_key=API_key)
    messages = [{"role": "user", "content": prompt_for_context_extraction.format(messages=messages_text)}]

    for _ in range(_MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                tools=CONTEXT_EXTRACTION_TOOLS,
                temperature=0.3,
            )
        except Exception as e:
            print(f"Ошибка при извлечении контекста: {e}")
            return

        choice = response.choices[0].message
        if not choice.tool_calls:
            return

        messages.append({
            "role": "assistant",
            "content": choice.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.tool_calls
            ],
        })

        for tool_call in choice.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                execute_fn(tool_call.function.name, args)
                result = "Записано"
            except Exception as e:
                result = f"Ошибка: {e}"
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})


@observe(as_type="generation", name="groq-image-caption", capture_input=False, capture_output=False)
def caption_image(image_b64):
    if not API_key:
        print("Ошибка: GROQ_API_KEY не установлен")
        return None

    client = Groq(api_key=API_key)
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_for_image_caption},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # This vision model is a reasoning model that prepends its
        # chain-of-thought in a <think> block -- only the text after it is
        # the actual caption.
        caption = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip() if content else None
        usage = response.usage
        langfuse.update_current_generation(
            model=VISION_MODEL,
            input=prompt_for_image_caption,
            output=caption,
            model_parameters={"temperature": 0.3},
            usage_details={
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens,
            } if usage else None,
        )
        return caption
    except Exception as e:
        print(f"Ошибка при описании изображения: {e}")
        return None

import os
import re

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

from config import get_groq_api_key
from llm.prompt import prompt_for_image_caption

API_KEY = get_groq_api_key()

PRIMARY_MODEL = "openai/gpt-oss-120b"
FAST_MODEL = "openai/gpt-oss-20b"
CONTEXT_LEARNING_MODEL = "qwen/qwen3.8-27b"
VISION_MODEL = "qwen/qwen3.6-27b"


def tracing_config():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return {}
    return {"callbacks": [CallbackHandler()]}


def get_chat_model(kind="primary", temperature=0.3):
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY не установлен")
    models = {
        "primary": PRIMARY_MODEL,
        "fast": FAST_MODEL,
        "context": CONTEXT_LEARNING_MODEL,
        "vision": VISION_MODEL,
    }
    model = ChatGroq(model=models[kind], api_key=API_KEY, temperature=temperature, max_retries=2)
    if kind == "primary":
        fallback = ChatGroq(model=FAST_MODEL, api_key=API_KEY, temperature=temperature, max_retries=2)
        return model.with_fallbacks([fallback])
    return model


def _content(message):
    return message.content if isinstance(message.content, str) else ""


def answer_context_question(full_prompt):
    try:
        return _content(get_chat_model("context", 0.5).invoke(full_prompt, config=tracing_config()))
    except Exception as e:
        print(f"Ошибка при контекстном запросе: {e}")
        return None


def caption_image(image_b64):
    try:
        message = HumanMessage(content=[
            {"type": "text", "text": prompt_for_image_caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ])
        content = _content(get_chat_model("vision", 0.3).invoke([message], config=tracing_config()))
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() or None
    except Exception as e:
        print(f"Ошибка при описании изображения: {e}")
        return None

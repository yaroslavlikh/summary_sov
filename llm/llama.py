import ollama
from llm.prompt import prompt_for_llm

def send_promte(prompt: str, model: str = "llama3"):
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt_for_llm + prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка LLM: {str(e)}"
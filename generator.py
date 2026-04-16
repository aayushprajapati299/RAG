"""
generator.py

Builds a prompt from retrieved pages + full chat history,
then calls the Groq API to generate a grounded answer.
"""

import os
import requests
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv() 

# Now os.environ.get will work correctly
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def build_prompt(question: str, pages: list, chat_history: list) -> str:
    """
    Builds a prompt that includes retrieved page context AND
    the previous conversation so the LLM can answer follow-ups.
    """
    prompt = """You are a smart and helpful document assistant.
You have been given context pages from a document and the conversation history so far.
Use both to answer the latest question accurately.

Rules:
- Use the document context as your primary source of facts
- Use the conversation history to understand follow-up questions
- Think step by step when comparing, ranking, or calculating
- If the answer is not in the context, say "I could not find this in the document"
- Do not make anything up

DOCUMENT CONTEXT:
"""
    for page in pages:
        page_num = page.get("page_number", "?")
        text = page.get("text", "")
        if len(text) > 800:
            text = text[:800] + "..."
        prompt += f"[Page {page_num}]: {text}\n\n"

    if chat_history:
        prompt += "\nCONVERSATION SO FAR:\n"
        for turn in chat_history:
            prompt += f"User: {turn['question']}\n"
            prompt += f"Assistant: {turn['answer']}\n\n"

    prompt += f"\nLATEST QUESTION:\n{question}\n\nANSWER (think step by step):\n"
    return prompt


def generate_answer(question: str, pages: list, chat_history: list = []) -> dict:
    """
    Calls the Groq LLM API with context + chat history.
    Returns answer text and source page numbers.
    """
    prompt_string = build_prompt(question, pages, chat_history)
    source_pages = [page.get("page_number") for page in pages if "page_number" in page]

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful document assistant with memory of the conversation."},
                {"role": "user", "content": prompt_string}
            ],
            "temperature": 0.4,
            "max_tokens": 1024
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"]
        return {"answer": answer_text, "sources": source_pages}

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }
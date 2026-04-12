import os
import requests

# Don't forget to export GROQ_API_KEY=your_key_here in your terminal before running this!
# If you're on Windows, just use 'set' instead of 'export'.
# Or honestly, just grab a free one from console.groq.com if you need it.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def build_prompt(question: str, pages: list) -> str:
    """
    Mashes the question and our retrieved pages together into a nice prompt for the LLM.
    We're giving it strict rules so it doesn't hallucinate too much.
    """
    prompt = """You are a smart and helpful assistant. You have been given context pages from a document.
Your job is to answer the question by carefully reading the context and reasoning from it.

Rules:
- Use the context to find relevant data, then think step by step to answer
- If the question asks for comparisons, rankings, or calculations, do them using the data in the context
- If the answer requires picking the best option from a list, do so and explain why
- Only if the information is completely absent from the context, say "I could not find this in the document"
- Do not make up data that is not in the context

CONTEXT:
"""

    for page in pages:
        page_num = page.get("page_number", "?")
        text = page.get("text", "")

        # Chop it off at 800 chars so we don't blow up the context window
        if len(text) > 800:
            text = text[:800] + "..."

        prompt += f"[Page {page_num}]: {text}\n\n"

    prompt += f"\nQUESTION:\n{question}\n\nANSWER (think step by step):\n"
    return prompt
    
    for page in pages:
        page_num = page.get("page_number", "?")
        text = page.get("text", "")
        
        # Just trimming this down so we keep the prompt a reasonable size
        if len(text) > 800:
            text = text[:800] + "..."
            
        prompt += f"[Page {page_num}]: {text}\n"
        
    prompt += f"\nQUESTION:\n{question}\n\nANSWER:\n"
    return prompt

def generate_answer(question: str, pages: list) -> dict:
    """
    Wires up the prompt and shoots it over to the Groq API.
    Returns what it said, along with the source pages we used.
    Also tries to catch errors so it doesn't just crash out.
    """
    prompt_string = build_prompt(question, pages)
    
    # Grab just the page numbers so we can show where we got our answer from
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
                {"role": "system", "content": "You are a helpful document assistant."},
                {"role": "user", "content": prompt_string}
            ],
            "temperature": 0.4,   
            "max_tokens": 1024,   
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        # Bail out if the API yells at us (like a 401 or 500)
        response.raise_for_status() 
        
        # Dig through the JSON to pluck out the actual answer text
        data = response.json()
        answer_text = data["choices"][0]["message"]["content"]
        
        return {
            "answer": answer_text,
            "sources": source_pages
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": []
        }

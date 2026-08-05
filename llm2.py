from ollama import chat

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """
You are a senior MSME credit officer.

Rules:
- Never invent facts.
- ONLY use the information provided.
- If information is missing, explicitly state that.
- Write in a professional credit assessment style.
"""

def ask_llm(prompt):
    print(">>> USING OLLAMA <<<")
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content
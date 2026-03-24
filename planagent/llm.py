import os
import litellm
from dotenv import load_dotenv

load_dotenv()

# --- THE ONLY LINE YOU CHANGE TO SWITCH MODELS --- 
MODEL = "ollama/llama3.1" # phase 1: local dev, change as per requirements

# For ollama, point liteLLM at local server
if MODEL.startswith("ollama"):
    litellm.api_base = "http://localhost:11434"

def chat(messages:list,stream:bool = True) -> str:
    """
    messages formate:[{"role":"system"|"user"|"assistant","content":"..."}]
    """
    response =litellm.completion(
        model=MODEL,
        messages=messages,
        stream=stream
    )
    if stream:
        full = ""
        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            print(token,end="",flush=True)
            full += token
        print() # newline after stream ends
        return full
    else:
        return response.choices[0].message.content

def build_messages(system:str,history:list,user:str) -> list:
    """
    builds the messages array for any LLM call.
    """
    return [
        {"role":"system","content":system},
        *history,
        {"role":"user","content":user}
    ]
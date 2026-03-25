import os
import sys
import litellm
from dotenv import load_dotenv

load_dotenv()

# --- THE ONLY LINE YOU CHANGE TO SWITCH MODELS --- 
MODEL = "ollama/llama3.1" # phase 1: local dev, change as per requirements

# For ollama, point liteLLM at local server
if MODEL.startswith("ollama"):
    litellm.api_base = "http://localhost:11434"


class TokenTracker:
    """Tracks token usage across all LLM calls."""
    def __init__(self):
        self.calls = []          # per-call records
        self.total_input = 0
        self.total_output = 0

    def record(self, label: str, input_tokens: int, output_tokens: int):
        entry = {
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        self.calls.append(entry)
        self.total_input += input_tokens
        self.total_output += output_tokens
        return entry

    @property
    def total(self) -> int:
        return self.total_input + self.total_output

    def summary(self) -> dict:
        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total,
            "calls": self.calls,
        }


# Module-level tracker — shared across all calls
tracker = TokenTracker()


def _count_tokens(messages: list) -> int:
    """Count tokens in a message list using litellm."""
    try:
        return litellm.token_counter(model=MODEL, messages=messages)
    except Exception:
        # Fallback: rough estimate ~4 chars per token
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4


def _count_text_tokens(text: str) -> int:
    """Count tokens in a plain text string."""
    try:
        return litellm.token_counter(
            model=MODEL,
            messages=[{"role": "user", "content": text}]
        )
    except Exception:
        return len(text) // 4


def chat(messages: list, stream: bool = True, label: str = "chat") -> str:
    """
    messages formate:[{"role":"system"|"user"|"assistant","content":"..."}]
    Returns the response text. Token usage is recorded in `tracker`.
    """
    input_tokens = _count_tokens(messages)

    response = litellm.completion(
        model=MODEL,
        messages=messages,
        stream=stream
    )
    if stream:
        full = ""
        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            sys.stdout.write(token)
            sys.stdout.flush()
            full += token
        output_tokens = _count_text_tokens(full)
        tracker.record(label, input_tokens, output_tokens)
        return full
    else:
        content = response.choices[0].message.content
        output_tokens = _count_text_tokens(content)
        tracker.record(label, input_tokens, output_tokens)
        return content

def build_messages(system:str,history:list,user:str) -> list:
    """
    builds the messages array for any LLM call.
    """
    return [
        {"role":"system","content":system},
        *history,
        {"role":"user","content":user}
    ]
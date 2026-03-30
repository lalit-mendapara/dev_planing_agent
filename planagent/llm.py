import os
import sys
import hashlib
import json
import litellm
from dotenv import load_dotenv

load_dotenv()

# --- THE ONLY LINE YOU CHANGE TO SWITCH MODELS ---
MODEL = "ollama/gpt-oss:120b-cloud"  # phase 1: local dev, change as per requirements

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

# ---------------------------------------------------------------------------
# Response cache — avoid duplicate LLM calls for identical prompts
# ---------------------------------------------------------------------------
_response_cache: dict[str, str] = {}


def _cache_key(messages: list) -> str:
    """Deterministic hash of a message list for caching."""
    raw = json.dumps(messages, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main chat function
# ---------------------------------------------------------------------------

def chat(messages: list, stream: bool = True, label: str = "chat",
         json_mode: bool = False, use_cache: bool = False,
         stream_callback=None) -> str:
    """
    messages format: [{"role":"system"|"user"|"assistant","content":"..."}]
    Returns the response text. Token usage is recorded in `tracker`.

    json_mode: if True, requests structured JSON output from the model.
    use_cache: if True, returns cached response for identical prompts.
    stream_callback: optional callable(token: str) invoked for each streamed
        token.  When None the tokens are written to sys.stdout (legacy).
    """
    # Check cache for non-streaming calls
    if use_cache and not stream:
        key = _cache_key(messages)
        if key in _response_cache:
            cached = _response_cache[key]
            # Record as zero-cost cached call
            tracker.record(f"{label}_cached", 0, 0)
            return cached

    input_tokens = _count_tokens(messages)

    # Build extra kwargs
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = litellm.completion(**kwargs)

    if stream:
        full = ""
        for chunk in response:
            token = chunk.choices[0].delta.content or ""
            if stream_callback is not None:
                stream_callback(token)
            else:
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
        # Store in cache
        if use_cache:
            _response_cache[_cache_key(messages)] = content
        return content


def build_messages(system: str, history: list, user: str) -> list:
    """
    builds the messages array for any LLM call.
    """
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": user},
    ]
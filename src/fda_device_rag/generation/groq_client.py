"""Thin HTTP client for Groq's OpenAI-compatible chat completions API --
direct requests calls, not the groq SDK, matching this project's stance on
hand-rolling simple HTTP wrappers (see ollama_client.py's identical
rationale) even though Groq's own docs recommend their SDK.

Verified against Groq's docs before this module was written: base URL
https://api.groq.com/openai/v1, chat completions endpoint appends
/chat/completions, OpenAI-compatible request/response shape (model,
messages -> choices[0].message.content).

Unlike ollama_client.py, there is no pre-flight "is it ready" check here --
Groq is a hosted, generally-always-up API, so the "is the local server
running" failure mode that justifies Ollama's check_ollama_ready doesn't
apply. This module is deliberately just generate() and an error type.
"""
import requests

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"


class GroqError(Exception):
    """Raised when a Groq API call fails -- connection error, timeout,
    non-200 response, or an unexpected response shape."""


def generate(prompt: str, api_key: str, base_url: str = GROQ_BASE_URL, model: str = GROQ_MODEL, timeout: int = 60) -> str:
    """POSTs an OpenAI-compatible chat completion request. Any failure
    (timeout, connection drop, non-200 response, unexpected response
    shape) raises GroqError -- a single linear POST-and-unwrap with no
    branching worth unit-testing, same rationale as ollama_client.py's
    generate()."""
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError) as e:
        raise GroqError(f"Generation failed: {e}")

"""Thin HTTP client for a local Ollama server -- direct requests calls, not
the ollama package, matching this project's stance on hand-rolling simple
HTTP wrappers over adding a dependency for marginal convenience.

Verified against a live Ollama instance before this module was written:
GET /api/tags returns {"models": [{"name": "llama3:latest", ...}]}; POST
/api/generate (non-streaming) returns {"response": "...", "done": true, ...}.
"""
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "llama3"


class OllamaError(Exception):
    """Base class for generation-pipeline Ollama failures."""


class OllamaNotReadyError(OllamaError):
    """Raised by check_ollama_ready when the server is unreachable or the
    model isn't pulled -- a pre-flight failure, before any generation was
    attempted."""


class OllamaGenerationError(OllamaError):
    """Raised by generate() when the /api/generate call itself fails after
    the pre-flight check already passed -- kept distinct from
    OllamaNotReadyError so a mid-request failure (e.g. Ollama crashes
    between the check and the call) stays distinguishable from a pre-flight
    failure."""


def check_ollama_ready(base_url: str, model: str) -> None:
    """Raises OllamaNotReadyError with a specific message if the Ollama
    server isn't reachable (or returns an error status), or if it's
    reachable but `model` isn't pulled. Returns normally if ready."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        # Ollama's /api/tags returns tags with an explicit version, e.g.
        # "llama3:latest" -- match on the name before ':' so a bare "llama3"
        # check still matches whatever tag is actually pulled.
        pulled = {m["name"].split(":")[0] for m in response.json().get("models", [])}
    except (requests.RequestException, ValueError, KeyError):
        # ValueError covers response.json()'s JSONDecodeError (a subclass);
        # KeyError covers a models entry missing "name" -- either way, a
        # live-but-malformed response is just as not-ready as no response.
        raise OllamaNotReadyError("Ollama not running. Start it, then retry.")

    if model not in pulled:
        raise OllamaNotReadyError(f"Model {model!r} not found. Run: ollama pull {model}")


def generate(prompt: str, base_url: str, model: str, keep_alive: str = "30m", timeout: int = 60) -> str:
    """POSTs a non-streaming /api/generate request. Any failure (timeout,
    connection drop, non-200 response, unexpected response shape) raises
    OllamaGenerationError -- a single linear POST-and-unwrap with no
    branching worth unit-testing beyond this."""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "keep_alive": keep_alive},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
    except (requests.RequestException, KeyError) as e:
        raise OllamaGenerationError(f"Generation failed: {e}")

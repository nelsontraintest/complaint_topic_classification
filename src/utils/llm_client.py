"""
llm_client.py
─────────────
Wrapper around Ollama for local Gemma 4 inference.

Used for:
1. Auto-naming newly detected topic clusters (topic_labeler.py)
2. Optional: enriching generated complaint data

Prerequisites:
    1. Install Ollama: https://ollama.com/download
    2. Pull Gemma 4:   ollama pull gemma3:4b
    3. Start server:   ollama serve   (usually auto-starts)
"""

from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_LLM_CONFIG = CONFIG["llm"]


def ask_gemma(prompt: str, system: Optional[str] = None, verbose: bool = False) -> str:
    """
    Send a prompt to the local Gemma model via Ollama and return the response.

    Args:
        prompt:  The user message / question.
        system:  Optional system instruction for the model.
        verbose: If True, print the prompt and response.

    Returns:
        The model's text response.

    Raises:
        RuntimeError: If Ollama is not running or the model is not available.
    """
    try:
        import ollama
    except ImportError:
        raise ImportError("Install ollama: pip install ollama")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if verbose:
        print(f"\n🤖  Gemma prompt:\n{prompt}\n")

    try:
        response = ollama.chat(
            model=_LLM_CONFIG["model"],
            messages=messages,
            options={
                "temperature": _LLM_CONFIG["temperature"],
                "num_predict": _LLM_CONFIG["max_tokens"],
            },
        )
        result = response["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Ollama call failed: {e}\n"
            f"Make sure Ollama is running and '{_LLM_CONFIG['model']}' is pulled.\n"
            f"Run: ollama pull {_LLM_CONFIG['model']}"
        )

    if verbose:
        print(f"✅  Gemma response:\n{result}\n")

    return result


def is_ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def name_topic_cluster(sample_complaints: list[str]) -> dict:
    """
    Use Gemma 4 to generate a short topic name and description for a cluster of complaints.

    Args:
        sample_complaints: 5-10 representative complaints from the cluster.

    Returns:
        Dict with 'topic_name' (snake_case) and 'description'.
    """
    samples_formatted = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(sample_complaints[:8])
    )

    prompt = f"""You are a banking complaint analyst. Below are customer complaints that belong to the same category.

Complaints:
{samples_formatted}

Task:
1. Identify the ONE common banking problem in these complaints.
2. Give a SHORT topic name in snake_case (e.g., "virtual_card_issue", "biometric_auth_failure").
3. Give a ONE sentence description of this topic.

Respond in this exact format (nothing else):
TOPIC_NAME: <snake_case_name>
DESCRIPTION: <one sentence description>"""

    system = (
        "You are a concise banking complaint taxonomy expert. "
        "You respond only in the exact format requested. No explanations, no extra text."
    )

    response = ask_gemma(prompt, system=system)

    # Parse response
    topic_name = "unknown_topic"
    description = ""
    for line in response.splitlines():
        if line.startswith("TOPIC_NAME:"):
            topic_name = line.replace("TOPIC_NAME:", "").strip().lower().replace(" ", "_")
        elif line.startswith("DESCRIPTION:"):
            description = line.replace("DESCRIPTION:", "").strip()

    return {"topic_name": topic_name, "description": description}

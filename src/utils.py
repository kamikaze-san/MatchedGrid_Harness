"""Atomic IO and HarmBench loading utilities."""
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path


def atomic_append_jsonl(path: Path, record, as_line: bool = False):
    """Append a record to a JSONL file under an fcntl lock.

    Lock-then-write-then-fsync ensures that two concurrent processes (e.g. a re-submitted
    Slurm job that overlapped with the old one for a few seconds) can't interleave or
    truncate each other's writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", buffering=1) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            if as_line:
                f.write(str(record) + "\n")
            else:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: Path, text: str):
    """Write-then-rename pattern: avoids partial-file states on crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_harmbench(path: str, n: int, seed: int):
    """Load HarmBench prompts as a deterministic, seed-shuffled list of n items.

    Expected JSONL format: {"id": "...", "behavior": "...", "category": "..."}
    The 'id' field must be stable across seeds so resumption works.
    """
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure stable IDs even if upstream lacks them
            if "id" not in obj:
                h = hashlib.sha1(obj["behavior"].encode()).hexdigest()[:12]
                obj["id"] = f"hb_{h}"
            items.append(obj)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def vllm_chat(port: int, model: str, messages: list, max_tokens: int = 512,
              temperature: float = 1.0, top_p: float = 0.95, images=None, seed: int = None):
    """Send a chat request to a local vLLM OpenAI-compatible server.

    Lazy-imports requests so the runner module can be inspected without it installed.
    """
    import requests

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": _format_messages(messages, images),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _format_messages(messages, images):
    """Convert a simple list-of-{role, content} into OpenAI multimodal format if images given.

    images: list of (turn_idx, image_path_or_url) — attached to the user message at that turn.
    """
    if not images:
        return messages
    # Inject images into the matching user turn.
    out = []
    image_map = {idx: paths for idx, paths in images}
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and i in image_map:
            content = [{"type": "text", "text": msg["content"]}]
            for img_path in image_map[i]:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _file_to_data_url(img_path)},
                })
            out.append({"role": "user", "content": content})
        else:
            out.append(msg)
    return out


def _file_to_data_url(path: str) -> str:
    """Convert a local image file to a base64 data URL for vLLM."""
    import base64
    import mimetypes
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def safe_parse_json(text: str, default=None):
    """Best-effort JSON parsing from LLM output (handles ```json fences and trailing text)."""
    if not text:
        return default
    s = text.strip()
    # Strip code fences
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            s = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    # Find first { and matching }
    start = s.find("{")
    if start < 0:
        return default
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i+1])
                except json.JSONDecodeError:
                    return default
    return default

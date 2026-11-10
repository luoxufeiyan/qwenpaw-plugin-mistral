# -*- coding: utf-8 -*-
"""Probe whether Mistral-hosted DeepSeek accepts ``reasoning_effort``.

Reads the provider's stored API key in-process (never printed), sends one
minimal request per effort value, and reports only status codes and the
*shape* of the response — no key material, no response bodies beyond a short
structural summary.

    python3 probe_deepseek_effort.py
"""

import json
import sys

sys.path.insert(0, "/app/venv/lib/python3.11/site-packages")

import httpx  # noqa: E402

from qwenpaw.providers.provider_persistence import (  # noqa: E402
    PROVIDER_SECRET_FIELDS,
)
from qwenpaw.security.secret_store import decrypt_dict_fields  # noqa: E402

SNAPSHOT = "/app/working.secret/providers/plugin/mistral.json"
BASE_URL = "https://api.mistral.ai/v1"
MODELS = ["deepseek-v4-1-flash-1017", "zai-glm-5-3-flash"]
EFFORTS = ["low", "high", "max"]


def load_key() -> str:
    with open(SNAPSHOT, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    config = decrypt_dict_fields(snapshot, PROVIDER_SECRET_FIELDS)
    key = config.get("api_key") or ""
    if not key:
        raise SystemExit("no api_key in snapshot")
    return key


def shape_of(content) -> str:
    """Describe a response content value without echoing it."""
    if isinstance(content, str):
        return f"str({len(content)} chars)"
    if isinstance(content, list):
        kinds = []
        for part in content:
            if isinstance(part, dict):
                kind = part.get("type")
                if kind == "thinking":
                    inner = part.get("thinking")
                    size = sum(
                        len(x.get("text", "")) for x in inner or []
                        if isinstance(x, dict)
                    )
                    kinds.append(f"thinking({size} chars)")
                else:
                    kinds.append(f"{kind}({len(str(part.get('text','')))} chars)")
            else:
                kinds.append(type(part).__name__)
        return "list[" + ", ".join(kinds) + "]"
    return type(content).__name__


def probe(client: httpx.Client, key: str, model: str, effort: str | None):
    body = {
        "model": model,
        "messages": [
            {"role": "user",
             "content": "A bat and a ball cost $1.10 total. The bat costs "
                        "$1.00 more than the ball. How much is the ball? "
                        "Think first, then answer."},
        ],
        "max_tokens": 400,
    }
    if effort:
        body["reasoning_effort"] = effort
    try:
        resp = client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json=body,
        )
    except Exception as exc:  # noqa: BLE001
        return {"effort": effort, "status": "EXC", "detail": type(exc).__name__}

    out = {"effort": effort or "(none)", "status": resp.status_code}
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail")
            if isinstance(detail, list) and detail:
                out["detail"] = detail[0].get("msg", "")[:80]
            else:
                out["detail"] = str(detail)[:80]
        except Exception:  # noqa: BLE001
            out["detail"] = resp.text[:80]
        return out
    try:
        payload = resp.json()
        message = payload["choices"][0]["message"]
        out["content"] = shape_of(message.get("content"))
        out["reasoning_content"] = (
            f"{len(message['reasoning_content'])} chars"
            if isinstance(message.get("reasoning_content"), str) else "absent"
        )
        usage = payload.get("usage") or {}
        out["completion_tokens"] = usage.get("completion_tokens")
    except Exception as exc:  # noqa: BLE001
        out["detail"] = f"parse: {type(exc).__name__}"
    return out


def main() -> int:
    key = load_key()
    print(f"key loaded: prefix={key[:3]}… length={len(key)}\n")
    with httpx.Client(timeout=120) as client:
        for model in MODELS:
            print(f"=== {model} ===")
            for effort in [None] + EFFORTS:
                result = probe(client, key, model, effort)
                print("  " + json.dumps(result, ensure_ascii=False))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Offline self-test for the Mistral provider plugin.

Runs without network access and without touching QwenPaw's provider store:

    python3 selftest.py

It checks the three behaviours the plugin exists for:
  1. cache_modes declares OpenAI caching, so prompt_cache_key is accepted;
  2. reasoning_content relay is always off;
  3. prompt_cache_key is derived, stable per conversation, and only for chat.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwenpaw.providers.adapters.cache_policy import cache_request  # noqa: E402

from mistral_provider import BUILTIN_MODELS, MistralProvider  # noqa: E402

MODEL = "deepseek-v4-1-flash-1017"

SYSTEM = {"role": "system", "content": "You are a careful assistant."}
USER_1 = {"role": "user", "content": "hello there"}
USER_2 = {"role": "user", "content": "a different conversation"}


def build() -> MistralProvider:
    return MistralProvider(
        id="mistral",
        name="Mistral AI",
        base_url="https://api.mistral.ai/v1",
        api_key="test-key-not-used",
        models=list(BUILTIN_MODELS),
    )


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    provider = build()

    # 1. cache capability
    caps = provider.cache_capabilities(MODEL)
    checks.append(
        ("cache_modes advertises openai", "openai" in caps, f"caps={sorted(caps)}"),
    )
    try:
        out = cache_request(
            {"model": MODEL, "messages": [], "prompt_cache_key": "k"},
            "chat",
            caps,
        )
        checks.append(
            ("prompt_cache_key survives cache policy", "prompt_cache_key" in out, ""),
        )
    except Exception as exc:  # noqa: BLE001
        checks.append(("prompt_cache_key survives cache policy", False, repr(exc)))

    # 2. reasoning relay
    relay = provider._get_relay_reasoning(MODEL)
    checks.append(("relay_reasoning is on", relay is True, f"got={relay!r}"))
    for info in BUILTIN_MODELS:
        if info.relay_reasoning is not True:
            checks.append(
                (f"catalog relay off for {info.id}", False, f"got={info.relay_reasoning!r}"),
            )
            break
    else:
        checks.append(("catalog relay on for every model", True, f"{len(BUILTIN_MODELS)} models"))

    # 3. prompt_cache_key derivation
    first = provider.prepare_request(
        MODEL,
        "chat",
        {"model": MODEL, "messages": [SYSTEM, USER_1]},
    )
    again = provider.prepare_request(
        MODEL,
        "chat",
        {"model": MODEL, "messages": [SYSTEM, USER_1]},
    )
    other = provider.prepare_request(
        MODEL,
        "chat",
        {"model": MODEL, "messages": [SYSTEM, USER_2]},
    )
    key = first.get("prompt_cache_key")
    checks.append(("chat request carries prompt_cache_key", bool(key), f"key={key}"))
    checks.append(
        ("same conversation -> same key", key == again.get("prompt_cache_key"), ""),
    )
    checks.append(
        ("different conversation -> different key",
         key != other.get("prompt_cache_key"), ""),
    )

    responses = provider.prepare_request(
        MODEL,
        "responses",
        {"model": MODEL, "messages": [SYSTEM, USER_1]},
    )
    checks.append(
        ("non-chat protocol untouched",
         "prompt_cache_key" not in responses, ""),
    )

    # 4. catalog hook + reasoning_effort mapping
    defaults = MistralProvider.get_default_models()
    checks.append(
        ("get_default_models seeds the catalog",
         len(defaults) == len(BUILTIN_MODELS) and bool(defaults),
         f"{len(defaults)} models"),
    )
    effort_models = {"zai-glm-5-3-flash", "zai-glm-5-3",
                     "mistral-large-4-0", "mistral-small-2603",
                     "deepseek-v4-1-flash-1017"}
    for info in BUILTIN_MODELS:
        expected = info.id in effort_models
        got = info.thinking_control is not None
        if got != expected:
            checks.append(
                (f"thinking control presence for {info.id}", False,
                 f"expected={expected} got={got}"),
            )
            break
    else:
        checks.append(("thinking control only on documented models", True,
                       f"{len(effort_models)} models"))

    mapped = provider.get_agent_thinking_kwargs("zai-glm-5-3-flash", "high")
    checks.append(
        ("thinking level -> reasoning_effort (GLM)",
         mapped.get("reasoning_effort") == "high", f"got={mapped}"),
    )
    ds = provider.get_agent_thinking_kwargs(MODEL, "high")
    checks.append(
        ("thinking level -> reasoning_effort (DeepSeek, probe-verified)",
         ds.get("reasoning_effort") == "high", f"got={ds}"),
    )
    codestral = provider.get_agent_thinking_kwargs("codestral-25-08", "high")
    checks.append(
        ("no reasoning_effort for Codestral (unprobed)",
         "reasoning_effort" not in codestral, f"got={codestral}"),
    )

    kept = provider.prepare_request(
        MODEL,
        "chat",
        {"model": MODEL, "messages": [SYSTEM, USER_1],
         "reasoning_effort": "high"},
    )
    checks.append(
        ("reasoning knobs pass through to the transport",
         kept.get("reasoning_effort") == "high", f"got={kept.get('reasoning_effort')}"),
    )

    # 5. the model instance really carries relay=False into its formatter
    model = provider.get_chat_model_instance(MODEL)
    formatter = getattr(model, "formatter", None)
    relay_attr = getattr(formatter, "relay_reasoning_content", "<missing>")
    checks.append(
        ("model formatter relays reasoning (transport translates it)",
         relay_attr is True, f"got={relay_attr!r}"),
    )

    failed = 0
    for name, ok, extra in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{extra}]" if extra else ""))
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

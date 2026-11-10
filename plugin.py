# -*- coding: utf-8 -*-
"""Mistral AI provider plugin entry point."""

try:  # loaded as a package by the plugin loader
    from .mistral_provider import DEFAULT_BASE_URL, MistralProvider
except ImportError:  # loaded as a flat module
    from mistral_provider import DEFAULT_BASE_URL, MistralProvider


class MistralPlugin:
    """Registers the Mistral provider with QwenPaw's provider manager."""

    def register(self, api):
        api.register_provider(
            provider_id="mistral",
            provider_class=MistralProvider,
            label="Mistral AI",
            base_url=DEFAULT_BASE_URL,
            chat_model="OpenAIChatModel",
            require_api_key=True,
        )


plugin = MistralPlugin()

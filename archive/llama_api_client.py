"""Compatibility shim for legacy tests expecting a top-level llama_api_client module.

This module forwards to the real implementation in chronogrid.core.api_client.
It provides both LlamaAPIClient and the backwards-compatible LlamaApiClient alias.
"""

from chronogrid.core.api_client import (
    LlamaAPIClient as LlamaAPIClient,  # canonical name
    LlamaApiClient as LlamaApiClient,  # backwards-compatible alias
)

__all__ = ["LlamaAPIClient", "LlamaApiClient"]

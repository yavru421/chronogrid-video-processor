"""Minimal client for the Llama Vision chat completions API."""
from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable

import requests

logger = logging.getLogger(__name__)


class _ChatCompletions:
    def __init__(self, client: "LlamaAPIClient") -> None:
        self._client = client

    def create(self, **payload: Any) -> SimpleNamespace:
        response = None
        try:
            response = self._client.session.post(
                f"{self._client.base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            # Log error details but don't expose sensitive information
            logger.error(f"API request failed: {exc}")
            logger.debug(f"Request payload: {json.dumps(payload, indent=2)}")
            if response:
                logger.debug(f"Response content: {response.text}")
            raise  # Re-raise the exception
        except requests.RequestException as exc:
            logger.error(f"Network error during API call: {exc}")
            raise

        try:
            data: Dict[str, Any] = response.json()
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON response from API: {exc}")
            raise

        # Wrap in a simple namespace matching llama docs (completion_message.content.text)
        completion_message = data.get("completion_message")
        if completion_message is None and data.get("choices"):
            completion_message = data["choices"][0].get("message")

        text = ""
        if completion_message:
            content = completion_message.get("content")
            if isinstance(content, list):
                text = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            elif isinstance(content, dict):
                text = content.get("text", "")
            elif isinstance(content, str):
                text = content

        wrapped = SimpleNamespace(
            completion_message=SimpleNamespace(
                content=SimpleNamespace(text=text.strip()),
            ),
            raw=data,
        )
        return wrapped


class LlamaAPIClient:
    """Thin wrapper around the public Llama REST API."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.llama.com/v1") -> None:
        token = api_key or os.environ.get("LLAMA_API_KEY")
        if not token:
            raise ValueError("LLAMA_API_KEY environment variable is required")

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )
        self.chat = SimpleNamespace(completions=_ChatCompletions(self))

    def create_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Direct REST helper used by the legacy test suite."""
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def stream_chat_completion(self, payload: Dict[str, Any]) -> Iterable[str]:
        """Stream server-sent events from the chat completions endpoint."""
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=120,
            stream=True,
        )
        response.raise_for_status()
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    yield line
        finally:
            response.close()

    def upload_image_and_ask(self, *_: Any, **__: Any) -> Dict[str, Any]:
        raise NotImplementedError("upload_image_and_ask is not supported")

    def close(self) -> None:
        self.session.close()


class LlamaApiClient(LlamaAPIClient):
    """Backwards compatible alias exposing bearer_token kwarg."""

    def __init__(self, bearer_token: str | None = None, base_url: str = "https://api.llama.com/v1") -> None:
        super().__init__(api_key=bearer_token, base_url=base_url)

from __future__ import annotations

import asyncio
import json
import logging
import os
import typing as _t

from ..client import Client, ClientConfig, LoggingTracer

from .base import BaseLLM
from .registry import ModelRegistry

__all__ = [
    "GeminiLLM",
]

logger = logging.getLogger("codin.model.gemini_llm")


@ModelRegistry.register
class GeminiLLM(BaseLLM):
    """Implementation of BaseLLM for Google's Gemini API.
    
    Supports both streaming and non-streaming generation.
    
    Environment variables:
        LLM_PROVIDER: The LLM provider (should be 'gemini' for this class)
        LLM_API_KEY: The API key for the LLM provider
        LLM_BASE_URL: Base URL for the API (defaults to https://generativelanguage.googleapis.com)
        LLM_MODEL: The model to use (defaults to gemini-1.5-pro)
        
        Legacy environment variables (deprecated, will be removed):
        GOOGLE_API_KEY: The API key for Google AI
        GOOGLE_API_BASE: Base URL for the API
    """
    
    def __init__(self, model: str | None = None):
        # Get model from environment or use provided model or default
        env_model = os.environ.get("LLM_MODEL")
        model_name = model or env_model or "gemini-1.5-pro"
        
        super().__init__(model_name)
        
        # Try new environment variables first, fall back to legacy ones
        self.api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.api_base = os.environ.get("LLM_BASE_URL") or os.environ.get(
            "GOOGLE_API_BASE", "https://generativelanguage.googleapis.com"
        )
        
        # Remove trailing slash if present
        if self.api_base.endswith("/"):
            self.api_base = self.api_base[:-1]
        
        self._client: Client | None = None
        
        # Initialize the client
        if not self.api_key:
            raise ValueError("LLM_API_KEY or GOOGLE_API_KEY environment variable not set")
        
        # Configure the HTTP client
        config = ClientConfig(
            base_url=self.api_base,
            default_headers={
                "Content-Type": "application/json",
            },
            timeout=60.0,
            # Add tracing in debug mode
            tracers=[LoggingTracer()] if logger.isEnabledFor(logging.DEBUG) else []
        )
        self._client = Client(config)
        
        logger.info(f"Using Gemini API at {self.api_base} with model {self.model}")
    
    @classmethod
    def supported_models(cls) -> list[str]:
        return [
            r"gemini-.*",
        ]
    
    async def _ensure_client(self) -> Client:
        """Ensure the HTTP client is prepared and return it."""
        if not self._client:
            raise RuntimeError("Failed to initialize Gemini client")
        
        # Make sure the client is prepared
        await self._client.prepare()
        return self._client
    
    def _prepare_messages(self, prompt: str | list[dict[str, str]]) -> list[dict]:
        """Convert prompt to Gemini message format.
        
        Args:
            prompt: Either a string or a list of message dictionaries
            
        Returns:
            List of messages in Gemini format
        """
        if isinstance(prompt, str):
            return [{"role": "user", "parts": [{"text": prompt}]}]
        
        # Map OpenAI roles to Gemini roles
        gemini_messages = []
        
        for msg in prompt:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Gemini doesn't have system messages, prepend to first user message
                if gemini_messages and gemini_messages[0]["role"] == "user":
                    gemini_messages[0]["parts"].insert(0, {"text": f"System: {content}\n\n"})
                else:
                    gemini_messages.append({
                        "role": "user",
                        "parts": [{"text": f"System: {content}\n\n"}]
                    })
            elif role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:
                logger.warning(f"Unsupported role: {role}, treating as user")
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
        
        return gemini_messages
    
    async def generate(self, 
                      prompt: str | list[dict[str, str]], 
                      *, 
                      stream: bool = False,
                      temperature: float | None = None,
                      max_tokens: int | None = None,
                      stop_sequences: list[str] | None = None,
                      ) -> _t.AsyncIterator[str] | str:
        """Generate text using Gemini API."""
        client = await self._ensure_client()
        
        # Convert prompt to Gemini format
        messages = self._prepare_messages(prompt)
        
        # Prepare request payload
        payload = {
            "contents": messages,
            "generationConfig": {}
        }
        
        # Add optional parameters
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        
        if max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        
        if stop_sequences:
            payload["generationConfig"]["stopSequences"] = stop_sequences
        
        # Add API key to URL
        endpoint = f"/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        # Add streaming parameter to URL if needed
        if stream:
            endpoint += "&alt=sse"
            return self._stream_response(client, endpoint, payload)
        else:
            return await self._complete_response(client, endpoint, payload)
    
    async def _complete_response(self, client: Client, endpoint: str, payload: dict) -> str:
        """Handle a complete (non-streaming) response."""
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract content from response
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                return ""
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            # Join all text parts
            text = ""
            for part in parts:
                if "text" in part:
                    text += part["text"]
            
            return text
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Gemini response: {e}")
            logger.debug(f"Response data: {data}")
            raise ValueError(f"Failed to parse Gemini response: {e}")
    
    async def _stream_response(self, client: Client, endpoint: str, payload: dict) -> _t.AsyncIterator[str]:
        """Handle a streaming response."""
        async def stream_generator():
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        
                        candidates = data.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            
                            for part in parts:
                                if "text" in part:
                                    yield part["text"]
                                    
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE line: {line}")
        
        return stream_generator()
    
    async def generate_with_tools(self,
                                prompt: str | list[dict[str, str]],
                                tools: list[dict],
                                *,
                                stream: bool = False,
                                temperature: float | None = None,
                                max_tokens: int | None = None,
                                ) -> dict | _t.AsyncIterator[dict]:
        """Generate text with function/tool calling capabilities.
        
        Note: This is a placeholder implementation as Gemini's tool calling
        API might differ from OpenAI's. This needs to be updated when Gemini's
        function calling API is fully documented.
        """
        raise NotImplementedError("Tool calling is not yet implemented for Gemini")
    
    async def close(self):
        """Close the client and release resources."""
        if self._client:
            await self._client.close()
            self._client = None
    
    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        if self._client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._client.close())
            except Exception:
                pass 
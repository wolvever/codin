"""Simplified prompt system with elegant A2A protocol support.

This module provides concise, elegant prompt templates with:
- Local/remote storage (fs:// or http:// endpoints)
- Template versions and dict-based conditions
- A2A protocol compliance
- Simple API: prompt_run(name, [version], [variables], [tools], stream=True/False)
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from datetime import datetime
import typing as _t

from pydantic import BaseModel, Field, ConfigDict

# A2A SDK imports
from a2a.types import (
    Message as A2AMessage,
    Role as A2ARole,
    TextPart as A2ATextPart,
    FilePart as A2AFilePart,
    DataPart as A2ADataPart,
)

try:
    from jinja2 import Template as _JinjaTemplate
    _HAS_JINJA = True
except ImportError:  # pragma: no cover
    _HAS_JINJA = False
    _JinjaTemplate = None

__all__ = [
    "PromptTemplate",
    "PromptVariant",
    "RenderedPrompt",
    "ModelOptions",
    "A2ARole",
    "A2ATextPart",
    "A2AFilePart",
    "A2ADataPart",
    "A2APart", 
    "A2AMessage",
    "PromptResponse",
    "ToolDefinition",
]

# -----------------------------------------------------------------------------
# A2A Protocol Support
# -----------------------------------------------------------------------------

# Re-export a2a types for backward compatibility
A2AMessage = A2AMessage

# Union type for all part types
A2APart = A2ATextPart | A2AFilePart | A2ADataPart


class PromptResponse(BaseModel):
    """A2A protocol response structure."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    message: A2AMessage | None = None
    streaming: bool = False
    content: str | _t.AsyncIterator[str] | None = None
    error: dict[str, _t.Any] | None = None


class ToolDefinition(BaseModel):
    """Tool definition for LLM function calling."""
    model_config = ConfigDict(frozen=True)
    
    name: str
    description: str
    parameters: dict[str, _t.Any]
    metadata: dict[str, _t.Any] | None = None


# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------

def _make_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:10]


def _default_version(text: str) -> str:
    """Generate a deterministic version hash from the prompt text."""
    return _make_hash(text)


class ModelOptions(BaseModel):
    """Model-specific options for LLM generation."""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    
    def to_dict(self) -> dict[str, _t.Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.dict().items() if v is not None}


class PromptVariant(BaseModel):
    """A single template variant with simple dict-based conditions."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str
    conditions: dict[str, _t.Any] | None = None
    model: str | None = None
    model_provider: str | None = None
    model_options: ModelOptions | None = None
    system_prompt: str | None = None
    messages: list[dict[str, _t.Any]] | None = None
    
    # Private field for compiled template
    _compiled: _t.Any = None
    
    def __init__(self, **data):
        """Initialize and compile template for performance."""
        super().__init__(**data)
        if _HAS_JINJA:
            self._compiled = _JinjaTemplate(self.text, autoescape=False)
        else:
            self._compiled = None
    
    def matches(self, conditions: dict[str, _t.Any]) -> bool:
        """Check if this variant matches the given conditions using simple dict matching."""
        if not self.conditions:
            return True  # No conditions = matches everything
        
        for key, required in self.conditions.items():
            actual = conditions.get(key)
            
            # Simple equality check with some smart matching
            if key in ("model_family", "model_provider"):
                # Prefix matching for model families/providers
                if actual and required:
                    if not str(actual).lower().startswith(str(required).lower()):
                        return False
            elif isinstance(required, (bool, int, float)):
                # Exact match for primitive types
                if actual != required:
                    return False
            elif isinstance(required, str) and actual:
                # String contains/startswith matching
                if required.lower() not in str(actual).lower():
                    return False
            elif required != actual:
                # Default exact match
                return False
        
        return True
    
    def score(self, conditions: dict[str, _t.Any]) -> int:
        """Score how well this variant matches conditions (higher = better)."""
        if not self.conditions:
            return 1  # Base score for no conditions
        
        score = 0
        for key, required in self.conditions.items():
            actual = conditions.get(key)
            if actual == required:
                score += 10  # Exact match
            elif key in ("model_family", "model_provider") and actual and required:
                if str(actual).lower().startswith(str(required).lower()):
                    score += 5  # Prefix match
        
        return score
    
    def render(self, **variables: _t.Any) -> str:
        """Render the template with variables."""
        if self._compiled:
            return self._compiled.render(**variables)
        else:
            # Simple string formatting fallback
            return self.text.format(**variables) if variables else self.text
    
    def render_system_prompt(self, **variables: _t.Any) -> str | None:
        """Render the system prompt with variables."""
        if not self.system_prompt:
            return None
        
        if _HAS_JINJA:
            template = _JinjaTemplate(self.system_prompt, autoescape=False)
            return template.render(**variables)
        else:
            return self.system_prompt.format(**variables) if variables else self.system_prompt
    
    def render_messages(self, **variables: _t.Any) -> list[dict[str, _t.Any]] | None:
        """Render the messages list with variables."""
        if not self.messages:
            return None
        
        rendered_messages = []
        for msg in self.messages:
            rendered_msg = {}
            for key, value in msg.items():
                if isinstance(value, str) and _HAS_JINJA:
                    template = _JinjaTemplate(value, autoescape=False)
                    rendered_msg[key] = template.render(**variables)
                elif isinstance(value, str):
                    rendered_msg[key] = value.format(**variables) if variables else value
                else:
                    rendered_msg[key] = value
            rendered_messages.append(rendered_msg)
        
        return rendered_messages


class PromptTemplate:
    """A prompt template with multiple variants and simple version management."""
    
    def __init__(self, name: str, version: str = "latest", 
                 variants: list[PromptVariant] | None = None,
                 metadata: dict[str, _t.Any] | None = None,
                 text: str | None = None):
        """Initialize a prompt template.
        
        Args:
            name: Template name
            version: Template version
            variants: List of template variants (optional)
            metadata: Template metadata (optional)
            text: Simple text content (creates default variant if provided)
        """
        self.name = name
        self.version = version
        self.variants = variants or []
        self.metadata = metadata
        
        # Backward compatibility: if text is provided, create a default variant
        if text:
            default_variant = PromptVariant(text=text)
            self.variants.insert(0, default_variant)
    
    def add_variant(self, variant: PromptVariant) -> None:
        """Add a variant to this template."""
        self.variants.append(variant)
    
    def get_best_variant(self, conditions: dict[str, _t.Any] | None = None) -> PromptVariant | None:
        """Get the best matching variant using simple scoring."""
        if not self.variants:
            return None
        
        if not conditions:
            return self.variants[0]  # Default to first variant
        
        # Score all matching variants
        matches = []
        for variant in self.variants:
            if variant.matches(conditions):
                score = variant.score(conditions)
                matches.append((variant, score))
        
        if not matches:
            return self.variants[0]  # Fallback to first variant
        
        # Return highest scoring variant
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]
    
    def render(self, conditions: dict[str, _t.Any] | None = None, **variables: _t.Any) -> "RenderedPrompt":
        """Render the template with the best matching variant."""
        variant = self.get_best_variant(conditions)
        if not variant:
            raise ValueError(f"No variants available for template '{self.name}'")
        
        rendered_text = variant.render(**variables)
        rendered_system = variant.render_system_prompt(**variables)
        rendered_messages = variant.render_messages(**variables)
        
        return RenderedPrompt(
            text=rendered_text,
            template=self,
            variant=variant,
            variables=variables,
            rendered_at=datetime.now(),
            system_prompt=rendered_system,
            messages=rendered_messages
        )
    
    @property
    def text(self) -> str:
        """Get the text of the first variant (for backward compatibility)."""
        if self.variants:
            return self.variants[0].text
        return ""


class RenderedPrompt(BaseModel):
    """A rendered prompt ready for LLM execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    text: str
    template: PromptTemplate
    variant: PromptVariant
    variables: dict[str, _t.Any]
    rendered_at: datetime
    system_prompt: str | None = None
    messages: list[dict[str, _t.Any]] | None = None
    
    def __str__(self) -> str:
        return self.text
    
    @property
    def model_options(self) -> dict[str, _t.Any]:
        """Get model options as dict."""
        return self.variant.model_options.to_dict() if self.variant.model_options else {}
    
    def to_messages(self) -> list[dict[str, _t.Any]]:
        """Convert to message format for LLM APIs."""
        if self.messages:
            return self.messages
        
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.text})
        return messages 
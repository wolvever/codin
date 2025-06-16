# Prompt System Design

## Overview

The Prompt System provides template management, rendering, and execution capabilities for the CoDIN platform. It supports versioned prompt templates, variable substitution, and integration with multiple LLM providers while maintaining A2A protocol compliance.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Prompt System                               │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ PromptTemplate  │    │ PromptRegistry  │    │  PromptEngine   │ │
│  │   (Template)    │    │  (Storage)      │    │  (Execution)    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ TemplateLoader  │    │ VariableResolver│    │ RenderContext   │ │
│  │   (Loading)     │    │ (Substitution)  │    │   (Context)     │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ StorageBackend  │    │ TemplateCache   │    │ PromptValidator │ │
│  │ (Persistence)   │    │   (Caching)     │    │ (Validation)    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Interfaces

### PromptTemplate

```python
@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata."""
    
    name: str
    version: str
    content: str
    variables: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def render(self, variables: Dict[str, Any] = None) -> str:
        """Render template with variable substitution."""
        if variables is None:
            variables = {}
        
        # Use Jinja2 for template rendering
        template = Template(self.content)
        return template.render(**variables)
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate that all required variables are provided."""
        missing = []
        for var in self.variables:
            if var not in variables:
                missing.append(var)
        return missing
    
    def get_variable_schema(self) -> Dict[str, Any]:
        """Get JSON schema for template variables."""
        properties = {}
        for var in self.variables:
            # Extract type hints from metadata if available
            var_info = self.metadata.get("variables", {}).get(var, {})
            properties[var] = {
                "type": var_info.get("type", "string"),
                "description": var_info.get("description", f"Variable {var}")
            }
        
        return {
            "type": "object",
            "properties": properties,
            "required": self.variables
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "content": self.content,
            "variables": self.variables,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            version=data["version"],
            content=data["content"],
            variables=data["variables"],
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )
```

### PromptRegistry

```python
class PromptRegistry:
    """Registry for managing prompt templates."""
    
    def __init__(self, storage_backend: StorageBackend = None):
        self.storage_backend = storage_backend or FileStorageBackend()
        self.cache: Dict[str, PromptTemplate] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
    
    async def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        template_key = f"{template.name}:{template.version}"
        
        # Validate template
        validation_errors = await self._validate_template(template)
        if validation_errors:
            raise PromptValidationError(f"Template validation failed: {validation_errors}")
        
        # Store in backend
        await self.storage_backend.store_template(template)
        
        # Update cache
        self.cache[template_key] = template
        self.cache_timestamps[template_key] = datetime.now()
    
    async def get_template(
        self, 
        name: str, 
        version: str = "latest"
    ) -> Optional[PromptTemplate]:
        """Get prompt template by name and version."""
        template_key = f"{name}:{version}"
        
        # Check cache first
        if await self._is_cached(template_key):
            return self.cache[template_key]
        
        # Load from storage
        if version == "latest":
            template = await self.storage_backend.get_latest_template(name)
        else:
            template = await self.storage_backend.get_template(name, version)
        
        # Update cache
        if template:
            self.cache[template_key] = template
            self.cache_timestamps[template_key] = datetime.now()
        
        return template
    
    async def list_templates(self, name_filter: str = None) -> List[PromptTemplate]:
        """List all available templates."""
        return await self.storage_backend.list_templates(name_filter)
    
    async def delete_template(self, name: str, version: str) -> bool:
        """Delete a prompt template."""
        template_key = f"{name}:{version}"
        
        # Remove from storage
        deleted = await self.storage_backend.delete_template(name, version)
        
        # Remove from cache
        if template_key in self.cache:
            del self.cache[template_key]
            del self.cache_timestamps[template_key]
        
        return deleted
    
    async def get_template_versions(self, name: str) -> List[str]:
        """Get all versions of a template."""
        return await self.storage_backend.get_template_versions(name)
    
    async def _validate_template(self, template: PromptTemplate) -> List[str]:
        """Validate template content and metadata."""
        errors = []
        
        # Check for valid Jinja2 syntax
        try:
            Template(template.content)
        except TemplateError as e:
            errors.append(f"Invalid template syntax: {str(e)}")
        
        # Check that declared variables are actually used
        template_ast = Template(template.content).parse()
        used_variables = meta.find_undeclared_variables(template_ast)
        
        for var in template.variables:
            if var not in used_variables:
                errors.append(f"Declared variable '{var}' not used in template")
        
        for var in used_variables:
            if var not in template.variables:
                errors.append(f"Undeclared variable '{var}' used in template")
        
        return errors
    
    async def _is_cached(self, template_key: str) -> bool:
        """Check if template is cached and not expired."""
        if template_key not in self.cache:
            return False
        
        cache_time = self.cache_timestamps.get(template_key)
        if not cache_time:
            return False
        
        age = (datetime.now() - cache_time).total_seconds()
        return age < self.cache_ttl
```

### PromptEngine

```python
class PromptEngine:
    """Engine for executing prompt templates with LLM integration."""
    
    def __init__(
        self,
        registry: PromptRegistry,
        llm_factory: LLMFactory = None
    ):
        self.registry = registry
        self.llm_factory = llm_factory or LLMFactory()
        self.execution_cache: Dict[str, Any] = {}
    
    async def execute(
        self,
        template_name: str,
        variables: Dict[str, Any] = None,
        llm_config: Dict[str, Any] = None,
        version: str = "latest",
        context: RenderContext = None
    ) -> PromptResult:
        """Execute prompt template with LLM."""
        variables = variables or {}
        llm_config = llm_config or {}
        
        # Get template
        template = await self.registry.get_template(template_name, version)
        if not template:
            raise PromptNotFoundError(f"Template not found: {template_name}:{version}")
        
        # Validate variables
        missing_vars = template.validate_variables(variables)
        if missing_vars:
            raise PromptValidationError(f"Missing variables: {missing_vars}")
        
        # Create render context
        if context is None:
            context = RenderContext(
                template_name=template_name,
                template_version=template.version,
                variables=variables,
                metadata=template.metadata
            )
        
        # Render template
        rendered_prompt = await self._render_with_context(template, variables, context)
        
        # Execute with LLM if configured
        if llm_config:
            llm_result = await self._execute_with_llm(rendered_prompt, llm_config)
        else:
            llm_result = None
        
        return PromptResult(
            template_name=template_name,
            template_version=template.version,
            rendered_prompt=rendered_prompt,
            variables=variables,
            llm_result=llm_result,
            context=context,
            execution_time=0.0  # TODO: Track execution time
        )
    
    async def _render_with_context(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        context: RenderContext
    ) -> str:
        """Render template with enhanced context."""
        # Add context variables
        enhanced_variables = variables.copy()
        enhanced_variables.update({
            "_context": context.to_dict(),
            "_template": {
                "name": template.name,
                "version": template.version,
                "metadata": template.metadata
            }
        })
        
        # Add utility functions
        enhanced_variables.update({
            "now": datetime.now,
            "uuid": lambda: str(uuid.uuid4()),
            "len": len,
            "str": str,
            "int": int,
            "float": float
        })
        
        return template.render(enhanced_variables)
    
    async def _execute_with_llm(
        self,
        prompt: str,
        llm_config: Dict[str, Any]
    ) -> LLMResult:
        """Execute rendered prompt with LLM."""
        llm = await self.llm_factory.create_llm(llm_config)
        
        # Convert prompt to messages format
        messages = [
            Message(
                role=Role.USER,
                content=[TextPart(text=prompt)]
            )
        ]
        
        # Execute with LLM
        response_messages = await llm.chat_completions(messages)
        
        return LLMResult(
            messages=response_messages,
            model=llm_config.get("model", "unknown"),
            tokens_used=0,  # TODO: Track token usage
            cost=0.0  # TODO: Calculate cost
        )
```

## Storage Backends

### FileStorageBackend

```python
class FileStorageBackend(StorageBackend):
    """File-based storage backend for prompt templates."""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path("prompts")
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store_template(self, template: PromptTemplate) -> None:
        """Store template to filesystem."""
        template_dir = self.base_path / template.name
        template_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = template_dir / f"{template.version}.json"
        
        async with aiofiles.open(template_file, 'w') as f:
            await f.write(json.dumps(template.to_dict(), indent=2))
    
    async def get_template(
        self, 
        name: str, 
        version: str
    ) -> Optional[PromptTemplate]:
        """Get template from filesystem."""
        template_file = self.base_path / name / f"{version}.json"
        
        if not template_file.exists():
            return None
        
        async with aiofiles.open(template_file, 'r') as f:
            data = json.loads(await f.read())
            return PromptTemplate.from_dict(data)
    
    async def get_latest_template(self, name: str) -> Optional[PromptTemplate]:
        """Get latest version of template."""
        template_dir = self.base_path / name
        
        if not template_dir.exists():
            return None
        
        # Find latest version based on semantic versioning
        version_files = list(template_dir.glob("*.json"))
        if not version_files:
            return None
        
        # Sort by modification time (latest first)
        latest_file = max(version_files, key=lambda f: f.stat().st_mtime)
        version = latest_file.stem
        
        return await self.get_template(name, version)
    
    async def list_templates(self, name_filter: str = None) -> List[PromptTemplate]:
        """List all templates."""
        templates = []
        
        for template_dir in self.base_path.iterdir():
            if not template_dir.is_dir():
                continue
            
            if name_filter and name_filter not in template_dir.name:
                continue
            
            for version_file in template_dir.glob("*.json"):
                template = await self.get_template(template_dir.name, version_file.stem)
                if template:
                    templates.append(template)
        
        return templates
    
    async def delete_template(self, name: str, version: str) -> bool:
        """Delete template from filesystem."""
        template_file = self.base_path / name / f"{version}.json"
        
        if template_file.exists():
            template_file.unlink()
            return True
        
        return False
    
    async def get_template_versions(self, name: str) -> List[str]:
        """Get all versions of a template."""
        template_dir = self.base_path / name
        
        if not template_dir.exists():
            return []
        
        versions = []
        for version_file in template_dir.glob("*.json"):
            versions.append(version_file.stem)
        
        return sorted(versions)
```

### HTTPStorageBackend

```python
class HTTPStorageBackend(StorageBackend):
    """HTTP-based storage backend for prompt templates."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def store_template(self, template: PromptTemplate) -> None:
        """Store template via HTTP API."""
        url = f"{self.base_url}/templates"
        
        response = await self.client.post(
            url,
            json=template.to_dict(),
            headers=self._get_headers()
        )
        
        if response.status_code not in (200, 201):
            raise PromptStorageError(f"Failed to store template: {response.status_code}")
    
    async def get_template(
        self, 
        name: str, 
        version: str
    ) -> Optional[PromptTemplate]:
        """Get template via HTTP API."""
        url = f"{self.base_url}/templates/{name}/{version}"
        
        response = await self.client.get(url, headers=self._get_headers())
        
        if response.status_code == 404:
            return None
        elif response.status_code == 200:
            data = response.json()
            return PromptTemplate.from_dict(data)
        else:
            raise PromptStorageError(f"Failed to get template: {response.status_code}")
    
    async def list_templates(self, name_filter: str = None) -> List[PromptTemplate]:
        """List templates via HTTP API."""
        url = f"{self.base_url}/templates"
        params = {}
        if name_filter:
            params["filter"] = name_filter
        
        response = await self.client.get(
            url, 
            params=params,
            headers=self._get_headers()
        )
        
        if response.status_code == 200:
            templates_data = response.json()
            return [PromptTemplate.from_dict(data) for data in templates_data]
        else:
            raise PromptStorageError(f"Failed to list templates: {response.status_code}")
```

## Template Features

### Variable Types

```python
class VariableType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    MESSAGE_HISTORY = "message_history"
    AGENT_CONTEXT = "agent_context"

@dataclass
class VariableDefinition:
    name: str
    type: VariableType
    description: str
    required: bool = True
    default: Any = None
    validation: Dict[str, Any] = field(default_factory=dict)
    
    def validate_value(self, value: Any) -> bool:
        """Validate variable value against definition."""
        if value is None and self.required:
            return False
        
        if value is None and not self.required:
            return True
        
        # Type validation
        if self.type == VariableType.STRING and not isinstance(value, str):
            return False
        elif self.type == VariableType.NUMBER and not isinstance(value, (int, float)):
            return False
        elif self.type == VariableType.BOOLEAN and not isinstance(value, bool):
            return False
        elif self.type == VariableType.ARRAY and not isinstance(value, list):
            return False
        elif self.type == VariableType.OBJECT and not isinstance(value, dict):
            return False
        
        # Additional validation rules
        for rule, constraint in self.validation.items():
            if rule == "min_length" and len(str(value)) < constraint:
                return False
            elif rule == "max_length" and len(str(value)) > constraint:
                return False
            elif rule == "pattern" and not re.match(constraint, str(value)):
                return False
        
        return True
```

### Template Helpers

```python
class TemplateHelpers:
    """Helper functions available in template rendering."""
    
    @staticmethod
    def format_message_history(messages: List[Message]) -> str:
        """Format message history for prompt inclusion."""
        formatted = []
        for message in messages:
            role = message.role.value.upper()
            content = message.content[0].text if message.content else ""
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    @staticmethod
    def extract_code_blocks(text: str, language: str = None) -> List[str]:
        """Extract code blocks from text."""
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if language:
            return [code for lang, code in matches if lang == language]
        else:
            return [code for _, code in matches]
    
    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """Estimate token count for text."""
        # Simplified token counting (replace with proper tokenizer)
        return len(text.split()) * 1.3  # Rough approximation
```

## Prompt Validation

### PromptValidator

```python
class PromptValidator:
    """Validates prompt templates for quality and compliance."""
    
    def __init__(self):
        self.quality_rules = [
            self._check_clear_instructions,
            self._check_variable_usage,
            self._check_length_limits,
            self._check_a2a_compliance
        ]
    
    async def validate_template(self, template: PromptTemplate) -> ValidationResult:
        """Validate template against all rules."""
        errors = []
        warnings = []
        
        for rule in self.quality_rules:
            rule_result = await rule(template)
            errors.extend(rule_result.errors)
            warnings.extend(rule_result.warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    async def _check_clear_instructions(self, template: PromptTemplate) -> ValidationResult:
        """Check for clear, actionable instructions."""
        errors = []
        warnings = []
        
        content = template.content.lower()
        
        # Check for imperative language
        imperative_patterns = [
            r'\b(please|kindly|could you|would you)\b',
            r'\b(try to|attempt to|see if you can)\b'
        ]
        
        for pattern in imperative_patterns:
            if re.search(pattern, content):
                warnings.append(f"Consider using more direct instructions instead of: {pattern}")
        
        # Check for specific task description
        if not re.search(r'\b(analyze|create|generate|write|implement|fix)\b', content):
            warnings.append("Consider including specific action verbs for clarity")
        
        return ValidationResult(errors=errors, warnings=warnings)
    
    async def _check_variable_usage(self, template: PromptTemplate) -> ValidationResult:
        """Check proper variable usage."""
        errors = []
        warnings = []
        
        # Parse template to find used variables
        template_ast = Template(template.content).parse()
        used_variables = meta.find_undeclared_variables(template_ast)
        
        # Check for unused declared variables
        for var in template.variables:
            if var not in used_variables:
                warnings.append(f"Declared variable '{var}' is not used in template")
        
        # Check for undeclared used variables
        for var in used_variables:
            if var not in template.variables and not var.startswith('_'):
                errors.append(f"Undeclared variable '{var}' used in template")
        
        return ValidationResult(errors=errors, warnings=warnings)
    
    async def _check_length_limits(self, template: PromptTemplate) -> ValidationResult:
        """Check template length limits."""
        errors = []
        warnings = []
        
        content_length = len(template.content)
        
        if content_length > 8000:  # Typical context limit consideration
            warnings.append(f"Template is very long ({content_length} chars), consider breaking into smaller parts")
        
        if content_length < 50:
            warnings.append("Template might be too short to provide clear instructions")
        
        return ValidationResult(errors=errors, warnings=warnings)
    
    async def _check_a2a_compliance(self, template: PromptTemplate) -> ValidationResult:
        """Check A2A protocol compliance."""
        errors = []
        warnings = []
        
        content = template.content.lower()
        
        # Check for proper role definitions
        if 'assistant' not in content and 'you are' not in content:
            warnings.append("Consider defining the assistant's role clearly")
        
        # Check for output format specification
        if 'format' not in content and 'structure' not in content:
            warnings.append("Consider specifying expected output format")
        
        return ValidationResult(errors=errors, warnings=warnings)
```

## Configuration

### Prompt System Configuration

```python
@dataclass
class PromptConfig:
    storage_backend: str = "file"
    storage_path: str = "prompts"
    storage_url: Optional[str] = None
    storage_api_key: Optional[str] = None
    cache_ttl: int = 300
    max_template_size: int = 50000
    enable_validation: bool = True
    auto_backup: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptConfig":
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
```

## Error Handling

### Exception Types

```python
class PromptError(Exception):
    """Base exception for prompt system errors."""
    pass

class PromptNotFoundError(PromptError):
    """Raised when prompt template is not found."""
    pass

class PromptValidationError(PromptError):
    """Raised when prompt validation fails."""
    pass

class PromptStorageError(PromptError):
    """Raised when prompt storage operation fails."""
    pass

class PromptRenderError(PromptError):
    """Raised when prompt rendering fails."""
    pass

class PromptExecutionError(PromptError):
    """Raised when prompt execution fails."""
    pass
```

## Usage Examples

### Basic Template Usage

```python
# Create template
template = PromptTemplate(
    name="code_review",
    version="1.0",
    content="""
You are a senior software engineer reviewing code.

Review the following {{language}} code:

```{{language}}
{{code}}
```

Provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement

Format your response as structured feedback.
""",
    variables=["language", "code"],
    metadata={
        "description": "Code review template",
        "category": "development",
        "variables": {
            "language": {"type": "string", "description": "Programming language"},
            "code": {"type": "string", "description": "Code to review"}
        }
    }
)

# Register template
await registry.register_template(template)

# Execute template
result = await engine.execute(
    "code_review",
    variables={
        "language": "python",
        "code": "def hello():\n    print('Hello, World!')"
    },
    llm_config={
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.3
    }
)

print(result.llm_result.messages[0].content[0].text)
```

This prompt system design provides a robust foundation for managing and executing prompt templates with versioning, validation, and multi-backend support while maintaining integration with the broader CoDIN platform.
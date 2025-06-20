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

### Example 1: Development Workflow Prompts

```python
import asyncio
from codin.prompt.base import PromptTemplate
from codin.prompt.registry import PromptRegistry
from codin.prompt.engine import PromptEngine
from codin.prompt.storage import FileStorageBackend
from codin.model.factory import LLMFactory

async def setup_development_prompts():
    """Set up prompt system with development-focused templates."""
    
    # Create storage backend and registry
    storage = FileStorageBackend(base_path="prompts/development")
    registry = PromptRegistry(storage_backend=storage)
    
    # Create prompt engine with LLM factory
    llm_factory = LLMFactory()
    engine = PromptEngine(registry=registry, llm_factory=llm_factory)
    
    # Code Review Template
    code_review_template = PromptTemplate(
        name="code_review",
        version="2.0",
        content="""You are an expert software engineer conducting a thorough code review.

**Code to Review:**
```{{language}}
{{code}}
```

**Review Context:**
- Project: {{project_name}}
- File: {{file_path}}
- Author: {{author}}
- Purpose: {{purpose}}

**Review Criteria:**
Evaluate the code against these criteria:

1. **Code Quality & Style**
   - Follows language conventions and best practices
   - Clear and descriptive naming
   - Appropriate code organization

2. **Functionality & Logic**
   - Correct implementation of requirements
   - Proper error handling
   - Edge case consideration

3. **Security & Performance**
   - No security vulnerabilities
   - Efficient algorithms and data structures
   - Resource usage optimization

4. **Maintainability**
   - Code readability and documentation
   - Testability and modularity
   - Future extensibility

**Output Format:**
Provide your review as structured feedback:

## Summary
Brief overall assessment (1-2 sentences)

## Issues Found
{{#if issues_found}}
### Critical Issues
- [List critical issues that must be fixed]

### Minor Issues  
- [List minor issues and suggestions]
{{/if}}

## Recommendations
- [Specific actionable recommendations]

## Code Quality Score: X/10
[Justification for the score]
""",
        variables=["language", "code", "project_name", "file_path", "author", "purpose"],
        metadata={
            "description": "Comprehensive code review template",
            "category": "development",
            "version": "2.0",
            "variables": {
                "language": {"type": "string", "description": "Programming language"},
                "code": {"type": "string", "description": "Code to review"},
                "project_name": {"type": "string", "description": "Project name"},
                "file_path": {"type": "string", "description": "File path"},
                "author": {"type": "string", "description": "Code author"},
                "purpose": {"type": "string", "description": "Code purpose/feature"}
            }
        }
    )
    
    # Architecture Design Template
    architecture_template = PromptTemplate(
        name="system_architecture",
        version="1.5",
        content="""You are a senior software architect designing a system architecture.

**Requirements:**
{{requirements}}

**Constraints:**
- Budget: {{budget}}
- Timeline: {{timeline}}
- Team size: {{team_size}}
- Technology preferences: {{tech_preferences}}
- Scalability requirements: {{scalability_needs}}

**Design Approach:**
Create a comprehensive system architecture that addresses:

1. **High-Level Architecture**
   - System components and their relationships
   - Data flow between components
   - Technology stack selection

2. **Detailed Design**
   - Database schema design
   - API design and endpoints
   - Security architecture
   - Deployment architecture

3. **Implementation Plan**
   - Development phases
   - Risk assessment
   - Performance considerations

**Output Format:**
## Architecture Overview
[High-level system description]

## System Components
### Component 1: [Name]
- **Purpose:** [Description]
- **Technology:** [Tech stack]
- **Interfaces:** [APIs/connections]

[Repeat for each component]

## Database Design
[Schema and relationships]

## API Specification
[Key endpoints and data formats]

## Security Architecture
[Authentication, authorization, data protection]

## Deployment Strategy
[Infrastructure and scaling approach]

## Implementation Roadmap
### Phase 1: [Foundation]
- [Tasks and deliverables]

### Phase 2: [Core Features]
- [Tasks and deliverables]

### Phase 3: [Advanced Features]
- [Tasks and deliverables]

## Risk Assessment
[Potential risks and mitigation strategies]
""",
        variables=["requirements", "budget", "timeline", "team_size", "tech_preferences", "scalability_needs"],
        metadata={
            "description": "System architecture design template",
            "category": "development",
            "variables": {
                "requirements": {"type": "string", "description": "System requirements"},
                "budget": {"type": "string", "description": "Budget constraints"},
                "timeline": {"type": "string", "description": "Timeline requirements"},
                "team_size": {"type": "string", "description": "Development team size"},
                "tech_preferences": {"type": "string", "description": "Technology preferences"},
                "scalability_needs": {"type": "string", "description": "Scalability requirements"}
            }
        }
    )
    
    # Test Generation Template
    test_generation_template = PromptTemplate(
        name="test_generation",
        version="1.0",
        content="""You are a test automation expert creating comprehensive test suites.

**Code to Test:**
```{{language}}
{{code}}
```

**Testing Context:**
- Function/Class: {{target_name}}
- Test Framework: {{test_framework}}
- Coverage Requirements: {{coverage_requirements}}
- Test Types Needed: {{test_types}}

**Test Generation Guidelines:**
Generate tests that cover:

1. **Happy Path Testing**
   - Normal input scenarios
   - Expected behavior validation

2. **Edge Case Testing**
   - Boundary conditions
   - Null/empty inputs
   - Maximum/minimum values

3. **Error Handling Testing**
   - Invalid inputs
   - Exception scenarios
   - Error message validation

4. **Integration Testing** (if applicable)
   - Component interactions
   - External dependencies

**Output Format:**
```{{language}}
{{#if test_framework == "pytest"}}
import pytest
from unittest.mock import Mock, patch
{{/if}}
{{#if test_framework == "jest"}}
const { {{target_name}} } = require('./{{module_name}}');
{{/if}}

# Test class/describe block
{{#if test_framework == "pytest"}}
class Test{{target_name}}:
{{/if}}
{{#if test_framework == "jest"}}
describe('{{target_name}}', () => {
{{/if}}
    
    # Happy path tests
    def test_{{target_name}}_normal_case(self):
        # Test normal operation
        pass
    
    # Edge case tests
    def test_{{target_name}}_edge_cases(self):
        # Test boundary conditions
        pass
    
    # Error handling tests
    def test_{{target_name}}_error_handling(self):
        # Test error scenarios
        pass
    
    # Performance tests (if needed)
    def test_{{target_name}}_performance(self):
        # Test performance requirements
        pass

{{#if test_framework == "jest"}}
});
{{/if}}
```

## Test Plan Summary
- **Total Tests:** [Number]
- **Coverage Areas:** [List of areas covered]
- **Test Data:** [Required test data/fixtures]
- **Dependencies:** [External dependencies to mock]
""",
        variables=["language", "code", "target_name", "test_framework", "coverage_requirements", "test_types"],
        metadata={
            "description": "Comprehensive test generation template",
            "category": "development",
            "variables": {
                "language": {"type": "string", "description": "Programming language"},
                "code": {"type": "string", "description": "Code to test"},
                "target_name": {"type": "string", "description": "Function/class name"},
                "test_framework": {"type": "string", "description": "Test framework (pytest, jest, etc.)"},
                "coverage_requirements": {"type": "string", "description": "Coverage requirements"},
                "test_types": {"type": "string", "description": "Types of tests needed"}
            }
        }
    )
    
    # Register all templates
    await registry.register_template(code_review_template)
    await registry.register_template(architecture_template)
    await registry.register_template(test_generation_template)
    
    return registry, engine

# Usage example
async def conduct_code_review():
    """Demonstrate code review using prompt templates."""
    
    registry, engine = await setup_development_prompts()
    
    # Code to review
    sample_code = '''
def user_login(username, password):
    user = db.query("SELECT * FROM users WHERE username = '" + username + "'")
    if user and user.password == password:
        session['user_id'] = user.id
        return {"success": True, "user_id": user.id}
    return {"success": False, "error": "Invalid credentials"}
'''
    
    # Execute code review
    review_result = await engine.execute(
        template_name="code_review",
        variables={
            "language": "python",
            "code": sample_code,
            "project_name": "UserAuth System",
            "file_path": "src/auth/login.py",
            "author": "junior_dev",
            "purpose": "User authentication endpoint"
        },
        llm_config={
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "temperature": 0.2
        }
    )
    
    print("🔍 Code Review Results:")
    print("=" * 50)
    print(review_result.llm_result.messages[0].content[0].text)

async def design_system_architecture():
    """Demonstrate architecture design using prompt templates."""
    
    registry, engine = await setup_development_prompts()
    
    # Execute architecture design
    arch_result = await engine.execute(
        template_name="system_architecture",
        variables={
            "requirements": """
Build a real-time chat application with:
- User authentication and profiles
- Public and private chat rooms
- Message history and search
- File sharing capabilities
- Mobile and web clients
- Support for 100,000 concurrent users
""",
            "budget": "$500,000",
            "timeline": "6 months",
            "team_size": "8 developers",
            "tech_preferences": "Python/FastAPI backend, React frontend, PostgreSQL",
            "scalability_needs": "Horizontal scaling, global deployment"
        },
        llm_config={
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.3
        }
    )
    
    print("🏗️ System Architecture Design:")
    print("=" * 50)
    print(arch_result.llm_result.messages[0].content[0].text)

async def generate_tests():
    """Demonstrate test generation using prompt templates."""
    
    registry, engine = await setup_development_prompts()
    
    # Code to test
    target_code = '''
class UserService:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def create_user(self, username, email, password):
        if not username or not email or not password:
            raise ValueError("All fields are required")
        
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        
        # Check if user exists
        existing = self.db.query("SELECT id FROM users WHERE username = ? OR email = ?", 
                               [username, email])
        if existing:
            raise ValueError("User already exists")
        
        # Hash password and create user
        hashed_password = hash_password(password)
        user_id = self.db.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            [username, email, hashed_password]
        )
        
        return {"id": user_id, "username": username, "email": email}
'''
    
    # Execute test generation
    test_result = await engine.execute(
        template_name="test_generation",
        variables={
            "language": "python",
            "code": target_code,
            "target_name": "UserService",
            "test_framework": "pytest",
            "coverage_requirements": "100% line coverage",
            "test_types": "unit tests, integration tests, error handling tests"
        },
        llm_config={
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "temperature": 0.1
        }
    )
    
    print("🧪 Generated Tests:")
    print("=" * 50)
    print(test_result.llm_result.messages[0].content[0].text)

if __name__ == "__main__":
    print("🚀 Prompt System Examples")
    print("=" * 50)
    
    # Run all examples
    asyncio.run(conduct_code_review())
    print("\n" + "=" * 50)
    asyncio.run(design_system_architecture())
    print("\n" + "=" * 50) 
    asyncio.run(generate_tests())
```

### Example 2: CLI Integration with Prompts

```python
#!/usr/bin/env python3
"""
CLI tool for CoDIN prompt system.

Usage:
    python prompt_cli.py review --file main.py --project myapp
    python prompt_cli.py architect --requirements requirements.txt
    python prompt_cli.py test --target UserService --file user_service.py
"""

import asyncio
import click
from pathlib import Path
import json

from codin.prompt.registry import PromptRegistry
from codin.prompt.engine import PromptEngine
from codin.model.factory import LLMFactory

class PromptCLI:
    """CLI interface for prompt system."""
    
    def __init__(self):
        self.registry = None
        self.engine = None
    
    async def initialize(self):
        """Initialize prompt system."""
        # Load prompts from development directory
        self.registry = PromptRegistry()
        await self.registry.from_config({
            "toolsets": {
                "development": {"endpoint": "fs://prompts/development"}
            }
        })
        
        self.engine = PromptEngine(
            registry=self.registry,
            llm_factory=LLMFactory()
        )

@click.group()
def cli():
    """CoDIN Prompt System CLI"""
    pass

@cli.command()
@click.option('--file', required=True, help='Code file to review')
@click.option('--project', required=True, help='Project name')
@click.option('--author', default='unknown', help='Code author')
@click.option('--model', default='gpt-4', help='LLM model to use')
def review(file, project, author, model):
    """Review code using prompt templates."""
    asyncio.run(_run_review(file, project, author, model))

@cli.command()
@click.option('--requirements', required=True, help='Requirements file or text')
@click.option('--budget', default='not specified', help='Budget constraints')
@click.option('--timeline', default='not specified', help='Timeline')
@click.option('--team-size', default='not specified', help='Team size')
@click.option('--model', default='gpt-4', help='LLM model to use')
def architect(requirements, budget, timeline, team_size, model):
    """Design system architecture."""
    asyncio.run(_run_architect(requirements, budget, timeline, team_size, model))

@cli.command()
@click.option('--target', required=True, help='Target class/function name')
@click.option('--file', required=True, help='Source code file')
@click.option('--framework', default='pytest', help='Test framework')
@click.option('--model', default='gpt-4', help='LLM model to use')
def test(target, file, framework, model):
    """Generate tests for code."""
    asyncio.run(_run_test(target, file, framework, model))

# CLI command implementations
async def _run_review(file_path, project, author, model):
    """Run code review."""
    cli_instance = PromptCLI()
    await cli_instance.initialize()
    
    try:
        # Read code file
        if not Path(file_path).exists():
            click.echo(f"❌ File not found: {file_path}", err=True)
            return
        
        with open(file_path, 'r') as f:
            code_content = f.read()
        
        # Detect language from file extension
        language = Path(file_path).suffix[1:]  # Remove the dot
        if language == 'py':
            language = 'python'
        elif language == 'js':
            language = 'javascript'
        
        # Execute review
        result = await cli_instance.engine.execute(
            template_name="code_review",
            variables={
                "language": language,
                "code": code_content,
                "project_name": project,
                "file_path": file_path,
                "author": author,
                "purpose": "Code review requested via CLI"
            },
            llm_config={
                "provider": "openai",
                "model": model,
                "temperature": 0.2
            }
        )
        
        click.echo("🔍 Code Review Results:")
        click.echo("=" * 50)
        click.echo(result.llm_result.messages[0].content[0].text)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

async def _run_architect(requirements_input, budget, timeline, team_size, model):
    """Run architecture design."""
    cli_instance = PromptCLI()
    await cli_instance.initialize()
    
    try:
        # Read requirements from file or use as text
        if Path(requirements_input).exists():
            with open(requirements_input, 'r') as f:
                requirements = f.read()
        else:
            requirements = requirements_input
        
        # Execute architecture design
        result = await cli_instance.engine.execute(
            template_name="system_architecture",
            variables={
                "requirements": requirements,
                "budget": budget,
                "timeline": timeline,
                "team_size": team_size,
                "tech_preferences": "Modern stack (determined by architect)",
                "scalability_needs": "Standard web application scaling"
            },
            llm_config={
                "provider": "openai",
                "model": model,
                "temperature": 0.3
            }
        )
        
        click.echo("🏗️ System Architecture Design:")
        click.echo("=" * 50)
        click.echo(result.llm_result.messages[0].content[0].text)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

async def _run_test(target_name, file_path, framework, model):
    """Run test generation."""
    cli_instance = PromptCLI()
    await cli_instance.initialize()
    
    try:
        # Read source file
        if not Path(file_path).exists():
            click.echo(f"❌ File not found: {file_path}", err=True)
            return
        
        with open(file_path, 'r') as f:
            code_content = f.read()
        
        # Detect language
        language = Path(file_path).suffix[1:]
        if language == 'py':
            language = 'python'
        elif language == 'js':
            language = 'javascript'
        
        # Execute test generation
        result = await cli_instance.engine.execute(
            template_name="test_generation",
            variables={
                "language": language,
                "code": code_content,
                "target_name": target_name,
                "test_framework": framework,
                "coverage_requirements": "Comprehensive coverage",
                "test_types": "unit, integration, edge cases"
            },
            llm_config={
                "provider": "openai",
                "model": model,
                "temperature": 0.1
            }
        )
        
        click.echo("🧪 Generated Tests:")
        click.echo("=" * 50)
        click.echo(result.llm_result.messages[0].content[0].text)
        
        # Optionally save to file
        test_file = f"test_{Path(file_path).stem}.py"
        save = click.confirm(f"Save tests to {test_file}?")
        if save:
            with open(test_file, 'w') as f:
                f.write(result.llm_result.messages[0].content[0].text)
            click.echo(f"✅ Tests saved to {test_file}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)

if __name__ == '__main__':
    cli()
```

### Example 3: Agent Integration with Prompts

```python
from codin.agent.base_agent import BaseAgent
from codin.prompt.registry import PromptRegistry
from codin.prompt.engine import PromptEngine

class PromptAwareAgent(BaseAgent):
    """Agent that uses prompt templates for structured responses."""
    
    def __init__(self, agent_id: str, prompt_registry: PromptRegistry, **kwargs):
        super().__init__(agent_id=agent_id, **kwargs)
        self.prompt_engine = PromptEngine(registry=prompt_registry)
    
    async def _use_template(self, template_name: str, variables: dict, context: str = ""):
        """Use a prompt template to generate structured response."""
        
        # Add context to variables
        enhanced_variables = variables.copy()
        enhanced_variables["agent_context"] = context
        enhanced_variables["agent_id"] = self.agent_id
        
        # Execute template
        result = await self.prompt_engine.execute(
            template_name=template_name,
            variables=enhanced_variables,
            llm_config={
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.3
            }
        )
        
        return result.llm_result.messages[0].content[0].text

# Usage with agent
async def create_code_review_agent():
    """Create an agent specialized in code review using prompts."""
    
    # Set up prompt registry
    registry = PromptRegistry()
    await registry.from_config({
        "toolsets": {
            "development": {"endpoint": "fs://prompts/development"}
        }
    })
    
    # Create agent
    agent = PromptAwareAgent(
        agent_id="code_reviewer",
        prompt_registry=registry,
        llm=LLMFactory.create_llm(),
        memory=MemMemoryService()
    )
    
    return agent

# Agent usage
async def automated_code_review():
    """Demonstrate automated code review with prompt-aware agent."""
    
    agent = await create_code_review_agent()
    
    # Code to review (could come from git hooks, CI/CD, etc.)
    code_changes = '''
def process_payment(amount, user_id, card_token):
    # Process payment
    if amount > 0:
        charge = stripe.charge.create(
            amount=amount * 100,  # Convert to cents
            currency='usd',
            source=card_token,
            description=f'Payment for user {user_id}'
        )
        return charge.id
    return None
'''
    
    # Use prompt template for structured review
    review = await agent._use_template(
        template_name="code_review",
        variables={
            "language": "python",
            "code": code_changes,
            "project_name": "E-commerce Platform",
            "file_path": "src/payments/processor.py",
            "author": "payment_dev",
            "purpose": "Payment processing functionality"
        },
        context="Automated review from CI/CD pipeline"
    )
    
    print("🤖 Automated Code Review:")
    print(review)

if __name__ == "__main__":
    asyncio.run(automated_code_review())
```

This prompt system design provides a robust foundation for managing and executing prompt templates with versioning, validation, and multi-backend support while maintaining integration with the broader CoDIN platform.
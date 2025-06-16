# Sandbox System Design

## Overview

The Sandbox System provides secure, isolated execution environments for code and shell commands within the CoDIN platform. It supports multiple backends including local execution, containerized environments, and cloud-based platforms while maintaining consistent APIs and safety controls.

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sandbox System                              │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │  Sandbox (ABC)  │    │  LocalSandbox   │    │  CodexSandbox   │ │
│  │                 │    │   (Process)     │    │   (Cloud)       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ SandboxFactory  │    │  E2BSandbox     │    │ DaytonaSandbox  │ │
│  │                 │    │ (Container)     │    │   (Remote)      │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│           │                       │                       │         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ SecurityPolicy  │    │ FileSystemOps   │    │ NetworkPolicy   │ │
│  │                 │    │                 │    │                 │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Interfaces

### Sandbox Protocol

```python
class Sandbox(ABC):
    """Abstract base class for sandbox environments."""
    
    @abstractmethod
    async def run(self, request: SandboxRequest) -> SandboxResult:
        """Execute code or command in sandbox."""
        pass
    
    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        """Write file to sandbox filesystem."""
        pass
    
    @abstractmethod
    async def read_file(self, path: str) -> str:
        """Read file from sandbox filesystem."""
        pass
    
    @abstractmethod
    async def list_files(self, path: str = ".") -> List[str]:
        """List files in sandbox directory."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        pass
```

### Request/Response Models

```python
@dataclass
class SandboxRequest:
    code: str
    language: str = "python"
    timeout: float = 30.0
    working_directory: str = "."
    environment: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)  # filename -> content
    network_enabled: bool = False
    max_memory: Optional[int] = None  # MB
    max_cpu_time: Optional[float] = None  # seconds

@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    memory_usage: Optional[int] = None  # MB
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    error: Optional[str] = None
```

## Implementation Details

### LocalSandbox

Executes code in local processes with security restrictions:

```python
class LocalSandbox(Sandbox):
    def __init__(
        self,
        base_path: Path = None,
        policy: SecurityPolicy = None
    ):
        self.base_path = base_path or Path("/tmp/codin_sandbox")
        self.policy = policy or DefaultSecurityPolicy()
        self.active_processes: Set[asyncio.subprocess.Process] = set()
    
    async def run(self, request: SandboxRequest) -> SandboxResult:
        """Execute code in local subprocess with restrictions."""
        # Create isolated working directory
        sandbox_dir = await self._create_sandbox_dir()
        
        try:
            # Write input files
            await self._write_input_files(sandbox_dir, request.files)
            
            # Prepare execution command
            cmd = await self._prepare_command(request, sandbox_dir)
            
            # Apply security policy
            cmd = await self.policy.apply_restrictions(cmd, request)
            
            # Execute with timeout and resource limits
            start_time = time.time()
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_dir,
                env=self._prepare_environment(request.environment)
            )
            
            self.active_processes.add(process)
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise SandboxTimeoutError(f"Execution timed out after {request.timeout}s")
            finally:
                self.active_processes.discard(process)
            
            execution_time = time.time() - start_time
            
            # Analyze filesystem changes
            files_created, files_modified = await self._analyze_fs_changes(sandbox_dir)
            
            return SandboxResult(
                stdout=stdout.decode('utf-8'),
                stderr=stderr.decode('utf-8'),
                exit_code=process.returncode,
                execution_time=execution_time,
                files_created=files_created,
                files_modified=files_modified
            )
            
        finally:
            # Cleanup sandbox directory
            await self._cleanup_sandbox_dir(sandbox_dir)
    
    async def _prepare_command(
        self, 
        request: SandboxRequest, 
        sandbox_dir: Path
    ) -> List[str]:
        """Prepare execution command based on language."""
        if request.language == "python":
            # Write code to temporary file
            code_file = sandbox_dir / "main.py"
            await self.write_file(str(code_file), request.code)
            return ["python", str(code_file)]
        
        elif request.language == "bash":
            # Write script to temporary file
            script_file = sandbox_dir / "script.sh"
            await self.write_file(str(script_file), request.code)
            return ["bash", str(script_file)]
        
        elif request.language == "javascript":
            code_file = sandbox_dir / "main.js"
            await self.write_file(str(code_file), request.code)
            return ["node", str(code_file)]
        
        else:
            raise UnsupportedLanguageError(f"Language not supported: {request.language}")
```

### CodexSandbox

Cloud-based execution through Codex service:

```python
class CodexSandbox(Sandbox):
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        session_timeout: float = 300.0
    ):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session_timeout = session_timeout
        self.session_client = httpx.AsyncClient()
    
    async def run(self, request: SandboxRequest) -> SandboxResult:
        """Execute code via Codex API."""
        payload = {
            "code": request.code,
            "language": request.language,
            "timeout": request.timeout,
            "environment": request.environment,
            "files": request.files,
            "network_enabled": request.network_enabled
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = await self.session_client.post(
                f"{self.api_endpoint}/execute",
                json=payload,
                headers=headers,
                timeout=request.timeout + 10.0  # Add buffer for network
            )
            
            if response.status_code != 200:
                raise SandboxExecutionError(
                    f"Codex API error: {response.status_code} - {response.text}"
                )
            
            result_data = response.json()
            
            return SandboxResult(
                stdout=result_data.get("stdout", ""),
                stderr=result_data.get("stderr", ""),
                exit_code=result_data.get("exit_code", 0),
                execution_time=result_data.get("execution_time", 0.0),
                memory_usage=result_data.get("memory_usage"),
                files_created=result_data.get("files_created", []),
                files_modified=result_data.get("files_modified", [])
            )
            
        except httpx.TimeoutException:
            raise SandboxTimeoutError("Codex API request timed out")
        except Exception as e:
            raise SandboxExecutionError(f"Codex execution failed: {str(e)}")
```

### E2BSandbox

Container-based execution using E2B platform:

```python
class E2BSandbox(Sandbox):
    def __init__(
        self,
        template_id: str = "base",
        api_key: str = None
    ):
        self.template_id = template_id
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.session: Optional[Session] = None
    
    async def run(self, request: SandboxRequest) -> SandboxResult:
        """Execute code in E2B container."""
        session = await self._get_or_create_session()
        
        try:
            # Upload files to session
            for filename, content in request.files.items():
                await session.filesystem.write(filename, content)
            
            # Execute code
            if request.language == "python":
                result = await session.run_python(
                    request.code,
                    timeout=request.timeout
                )
            elif request.language == "bash":
                result = await session.run_bash(
                    request.code,
                    timeout=request.timeout
                )
            else:
                # Write to file and execute
                filename = f"main.{self._get_file_extension(request.language)}"
                await session.filesystem.write(filename, request.code)
                
                cmd = self._get_run_command(request.language, filename)
                result = await session.run_bash(cmd, timeout=request.timeout)
            
            # Get filesystem changes
            files_created = await self._get_new_files(session)
            
            return SandboxResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                execution_time=result.execution_time,
                files_created=files_created,
                files_modified=[]  # E2B doesn't track modifications separately
            )
            
        finally:
            # Keep session alive for potential reuse
            pass
    
    async def _get_or_create_session(self) -> Session:
        """Get existing session or create new one."""
        if self.session is None or not await self.session.is_alive():
            self.session = await Session.create(
                template=self.template_id,
                api_key=self.api_key
            )
        return self.session
```

## Security Model

### Security Policy

```python
class SecurityPolicy(ABC):
    """Abstract security policy for sandbox execution."""
    
    @abstractmethod
    async def apply_restrictions(
        self, 
        command: List[str], 
        request: SandboxRequest
    ) -> List[str]:
        """Apply security restrictions to command."""
        pass
    
    @abstractmethod
    async def validate_code(self, code: str, language: str) -> bool:
        """Validate code for security issues."""
        pass

class DefaultSecurityPolicy(SecurityPolicy):
    def __init__(self):
        self.forbidden_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'open\s*\([^)]*["\'][wxa]["\']',  # Write mode file operations
        ]
        self.allowed_imports = {
            'math', 'json', 'datetime', 'random', 'string', 
            're', 'collections', 'itertools', 'functools'
        }
    
    async def apply_restrictions(
        self, 
        command: List[str], 
        request: SandboxRequest
    ) -> List[str]:
        """Apply resource and access restrictions."""
        restricted_cmd = ["timeout", str(request.timeout)]
        
        # Memory limit
        if request.max_memory:
            restricted_cmd.extend(["--memory", f"{request.max_memory}m"])
        
        # CPU time limit
        if request.max_cpu_time:
            restricted_cmd.extend(["--cpu-time", str(request.max_cpu_time)])
        
        # Disable network if required
        if not request.network_enabled:
            restricted_cmd.extend(["--network", "none"])
        
        restricted_cmd.extend(command)
        return restricted_cmd
    
    async def validate_code(self, code: str, language: str) -> bool:
        """Check code for security violations."""
        if language == "python":
            return await self._validate_python_code(code)
        elif language == "bash":
            return await self._validate_bash_code(code)
        return True
    
    async def _validate_python_code(self, code: str) -> bool:
        """Validate Python code for dangerous patterns."""
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        return True
```

### Filesystem Isolation

```python
class FileSystemManager:
    def __init__(self, sandbox_root: Path):
        self.sandbox_root = sandbox_root
        self.allowed_paths = {"/tmp", "/var/tmp"}
    
    async def create_isolated_directory(self, session_id: str) -> Path:
        """Create isolated directory for sandbox session."""
        session_dir = self.sandbox_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions
        session_dir.chmod(0o700)
        
        return session_dir
    
    async def validate_path(self, path: str, session_dir: Path) -> bool:
        """Validate that path is within allowed boundaries."""
        resolved_path = Path(path).resolve()
        
        # Check if path is within session directory
        try:
            resolved_path.relative_to(session_dir)
            return True
        except ValueError:
            return False
    
    async def safe_write_file(
        self, 
        path: str, 
        content: str, 
        session_dir: Path
    ) -> None:
        """Safely write file within sandbox boundaries."""
        if not await self.validate_path(path, session_dir):
            raise SecurityError(f"Path outside sandbox: {path}")
        
        file_path = session_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)
```

## Resource Management

### Resource Limits

```python
@dataclass
class ResourceLimits:
    max_memory_mb: int = 512
    max_cpu_time_seconds: float = 30.0
    max_wall_time_seconds: float = 60.0
    max_file_size_mb: int = 100
    max_files_count: int = 1000
    max_processes: int = 10
    max_network_connections: int = 0  # Disabled by default

class ResourceMonitor:
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.start_time: Optional[float] = None
        self.process: Optional[psutil.Process] = None
    
    async def start_monitoring(self, process: asyncio.subprocess.Process) -> None:
        """Start monitoring process resources."""
        self.start_time = time.time()
        self.process = psutil.Process(process.pid)
    
    async def check_limits(self) -> None:
        """Check if process exceeds resource limits."""
        if not self.process or not self.start_time:
            return
        
        # Check wall time
        wall_time = time.time() - self.start_time
        if wall_time > self.limits.max_wall_time_seconds:
            raise ResourceLimitError("Wall time limit exceeded")
        
        # Check memory usage
        try:
            memory_mb = self.process.memory_info().rss / (1024 * 1024)
            if memory_mb > self.limits.max_memory_mb:
                raise ResourceLimitError(f"Memory limit exceeded: {memory_mb}MB")
        except psutil.NoSuchProcess:
            # Process has terminated
            pass
        
        # Check CPU time
        try:
            cpu_times = self.process.cpu_times()
            cpu_time = cpu_times.user + cpu_times.system
            if cpu_time > self.limits.max_cpu_time_seconds:
                raise ResourceLimitError(f"CPU time limit exceeded: {cpu_time}s")
        except psutil.NoSuchProcess:
            pass
```

## Built-in Tools

### File Operations

```python
class SandboxFileTools:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
    
    async def search_files(
        self, 
        pattern: str, 
        directory: str = "."
    ) -> List[str]:
        """Search for files matching pattern."""
        files = await self.sandbox.list_files(directory)
        matching_files = []
        
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                matching_files.append(file)
        
        return matching_files
    
    async def grep_content(
        self, 
        pattern: str, 
        files: List[str] = None
    ) -> Dict[str, List[str]]:
        """Search for pattern in file contents."""
        if files is None:
            files = await self.sandbox.list_files()
        
        results = {}
        regex = re.compile(pattern)
        
        for file_path in files:
            try:
                content = await self.sandbox.read_file(file_path)
                matching_lines = []
                
                for line_num, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        matching_lines.append(f"{line_num}: {line}")
                
                if matching_lines:
                    results[file_path] = matching_lines
                    
            except Exception as e:
                # Skip files that can't be read
                continue
        
        return results
    
    async def edit_file(
        self, 
        path: str, 
        old_content: str, 
        new_content: str
    ) -> bool:
        """Edit file by replacing old content with new content."""
        try:
            current_content = await self.sandbox.read_file(path)
            
            if old_content not in current_content:
                return False
            
            updated_content = current_content.replace(old_content, new_content)
            await self.sandbox.write_file(path, updated_content)
            
            return True
            
        except Exception:
            return False
```

### Shell Operations

```python
class SandboxShellTools:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
    
    async def run_command(
        self, 
        command: str, 
        timeout: float = 30.0
    ) -> SandboxResult:
        """Run shell command in sandbox."""
        request = SandboxRequest(
            code=command,
            language="bash",
            timeout=timeout
        )
        return await self.sandbox.run(request)
    
    async def install_package(self, package: str, language: str = "python") -> SandboxResult:
        """Install package in sandbox."""
        if language == "python":
            command = f"pip install {package}"
        elif language == "javascript":
            command = f"npm install {package}"
        else:
            raise UnsupportedLanguageError(f"Package installation not supported for {language}")
        
        return await self.run_command(command, timeout=120.0)
```

## Factory Pattern

### Sandbox Factory

```python
class SandboxFactory:
    @staticmethod
    def create_sandbox(sandbox_type: str, **config) -> Sandbox:
        """Create sandbox instance based on type."""
        if sandbox_type == "local":
            return LocalSandbox(
                base_path=config.get("base_path"),
                policy=config.get("policy")
            )
        elif sandbox_type == "codex":
            return CodexSandbox(
                api_endpoint=config["api_endpoint"],
                api_key=config["api_key"]
            )
        elif sandbox_type == "e2b":
            return E2BSandbox(
                template_id=config.get("template_id", "base"),
                api_key=config.get("api_key")
            )
        elif sandbox_type == "daytona":
            return DaytonaSandbox(
                workspace_config=config["workspace_config"]
            )
        else:
            raise ValueError(f"Unknown sandbox type: {sandbox_type}")
    
    @staticmethod
    async def create_from_config(config: Dict[str, Any]) -> Sandbox:
        """Create sandbox from configuration dictionary."""
        sandbox_type = config.get("type", "local")
        return SandboxFactory.create_sandbox(sandbox_type, **config)
```

## Error Handling

### Exception Types

```python
class SandboxError(Exception):
    """Base exception for sandbox errors."""
    pass

class SandboxTimeoutError(SandboxError):
    """Raised when sandbox execution times out."""
    pass

class SandboxExecutionError(SandboxError):
    """Raised when sandbox execution fails."""
    pass

class ResourceLimitError(SandboxError):
    """Raised when resource limits are exceeded."""
    pass

class SecurityError(SandboxError):
    """Raised when security policy is violated."""
    pass

class UnsupportedLanguageError(SandboxError):
    """Raised when language is not supported."""
    pass
```

## Configuration

### Sandbox Configuration

```python
@dataclass
class SandboxConfig:
    type: str = "local"
    timeout: float = 30.0
    max_memory_mb: int = 512
    max_cpu_time: float = 30.0
    network_enabled: bool = False
    allowed_languages: List[str] = field(default_factory=lambda: ["python", "bash"])
    base_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    template_id: Optional[str] = None
    security_policy: str = "default"
```

## Performance Optimizations

### Session Reuse

- Keep sandbox sessions alive for multiple executions
- Pool sandbox instances for common configurations
- Lazy cleanup of idle sessions

### Caching

- Cache execution results for identical code
- Cache file system snapshots
- Cache container images and templates

### Parallelization

- Concurrent execution of independent sandbox operations
- Parallel file operations where possible
- Asynchronous resource monitoring

This sandbox design provides secure, flexible code execution capabilities with support for multiple backends and comprehensive safety controls.
"""Mock LLM implementation for testing and development.

This module provides a mock LLM that can generate reasonable responses
for coding tasks without requiring API keys.
"""

import asyncio
import json
import re
import typing as _t
from datetime import datetime

from .base import BaseLLM
from .registry import register


@register
class MockLLM(BaseLLM):
    """Mock LLM for testing that generates intelligent responses for coding tasks."""

    def __init__(self, model: str = "mock-llm"):
        """Initialize the mock LLM."""
        super().__init__(model)
        self.call_count = 0
        self._is_prepared = True  # Mock LLM is always prepared

    @classmethod
    def supported_models(cls) -> list[str]:
        """Return supported model patterns."""
        return ["mock-.*", "test-.*"]

    async def prepare(self, config=None):
        """Mock preparation - always ready."""
        pass

    def _extract_task_type(self, prompt: str) -> str:
        """Extract the type of coding task from the prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["fibonacci", "fib"]):
            return "fibonacci"
        elif any(word in prompt_lower for word in ["factorial"]):
            return "factorial"
        elif any(word in prompt_lower for word in ["sort", "bubble sort", "quick sort"]):
            return "sorting"
        elif any(word in prompt_lower for word in ["prime", "is_prime"]):
            return "prime"
        elif any(word in prompt_lower for word in ["calculator", "calc", "math"]):
            return "calculator"
        elif any(word in prompt_lower for word in ["hello world", "hello", "print"]):
            return "hello_world"
        elif any(word in prompt_lower for word in ["class", "object", "oop"]):
            return "class"
        elif any(word in prompt_lower for word in ["file", "read", "write"]):
            return "file_ops"
        elif any(word in prompt_lower for word in ["api", "request", "http"]):
            return "api_client"
        elif any(word in prompt_lower for word in ["database", "sql", "sqlite"]):
            return "database"
        else:
            return "general"

    def _generate_code_response(self, task_type: str, prompt: str) -> dict:
        """Generate appropriate code response based on task type."""
        responses = {
            "fibonacci": {
                "thinking": "I need to create a recursive Fibonacci function as requested.",
                "tool_calls": [
                    {
                        "name": "edit_file",
                        "arguments": {
                            "path": "fib.py",
                            "content": """def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number using recursion.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

if __name__ == "__main__":
    # Test with n=10
    result = fibonacci(10)
    print(f"fibonacci(10) = {result}")
""",
                            "explanation": "Creating the Fibonacci function file"
                        }
                    },
                    {
                        "name": "run_shell",
                        "arguments": {
                            "command": "python fib.py",
                            "explanation": "Testing the Fibonacci function"
                        }
                    }
                ],
                "message": "I'll create a recursive Fibonacci function and test it with n=10."
            },
            
            "factorial": {
                "thinking": "I need to create a factorial function, likely using recursion.",
                "tool_calls": [
                    {
                        "name": "edit_file",
                        "arguments": {
                            "path": "factorial.py",
                            "content": """def factorial(n):
    \"\"\"Calculate factorial of n using recursion.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)

if __name__ == "__main__":
    # Test factorial function
    for i in range(6):
        print(f"factorial({i}) = {factorial(i)}")
""",
                            "explanation": "Creating factorial function"
                        }
                    }
                ],
                "message": "I'll create a recursive factorial function for you."
            },
            
            "hello_world": {
                "thinking": "Creating a simple Hello World program.",
                "tool_calls": [
                    {
                        "name": "edit_file",
                        "arguments": {
                            "path": "hello.py",
                            "content": """#!/usr/bin/env python3
print("Hello, World!")
""",
                            "explanation": "Creating Hello World program"
                        }
                    },
                    {
                        "name": "run_shell",
                        "arguments": {
                            "command": "python hello.py",
                            "explanation": "Running the Hello World program"
                        }
                    }
                ],
                "message": "I'll create a Hello World program and run it for you."
            },
            
            "sorting": {
                "thinking": "Creating a sorting algorithm implementation.",
                "tool_calls": [
                    {
                        "name": "edit_file",
                        "arguments": {
                            "path": "sort.py",
                            "content": """def bubble_sort(arr):
    \"\"\"Sort array using bubble sort algorithm.\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def quick_sort(arr):
    \"\"\"Sort array using quick sort algorithm.\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_array}")
    print(f"Bubble sorted: {bubble_sort(test_array.copy())}")
    print(f"Quick sorted: {quick_sort(test_array.copy())}")
""",
                            "explanation": "Creating sorting algorithms"
                        }
                    }
                ],
                "message": "I'll create sorting algorithms (bubble sort and quick sort) for you."
            },
            
            "general": {
                "thinking": "This appears to be a general coding request. I'll create a basic Python program.",
                "tool_calls": [
                    {
                        "name": "edit_file",
                        "arguments": {
                            "path": "program.py",
                            "content": """#!/usr/bin/env python3
\"\"\"
General purpose Python program.
This is a template that can be customized for specific needs.
\"\"\"

def main():
    print("Program started successfully!")
    # Add your code here
    
if __name__ == "__main__":
    main()
""",
                            "explanation": "Creating a general Python program"
                        }
                    }
                ],
                "message": "I'll create a basic Python program template for you."
            }
        }
        
        return responses.get(task_type, responses["general"])

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate a response to the prompt."""
        self.call_count += 1
        
        # Convert prompt to string if it's a list
        if isinstance(prompt, list):
            prompt_text = " ".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict)])
        else:
            prompt_text = prompt
            
        # Extract task type and generate appropriate response
        task_type = self._extract_task_type(prompt_text)
        response_data = self._generate_code_response(task_type, prompt_text)
        
        # Add some metadata
        response_data.update({
            "should_continue": len(response_data.get("tool_calls", [])) > 0,
            "task_list": {
                "completed": [],
                "pending": [f"Execute {task_type} task"]
            }
        })
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        return json.dumps(response_data, indent=2)

    async def generate_with_tools(
        self,
        prompt: str | list[dict[str, str]],
        tools: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Generate with tools - delegate to generate for simplicity."""
        content = await self.generate(prompt, stream=stream, temperature=temperature, max_tokens=max_tokens)
        
        try:
            response_data = json.loads(content)
            return {
                "content": content,
                "tool_calls": response_data.get("tool_calls", [])
            }
        except json.JSONDecodeError:
            return {"content": content, "tool_calls": []}
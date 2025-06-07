#!/usr/bin/env python3
"""
Test script for prompt_run with Sealos AI Proxy API configuration
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add src to path so we can import codin modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from codin.prompt.run import prompt_run, set_endpoint


async def test_prompt_run_with_sealos():
    """Test prompt_run with the same Sealos AI Proxy configuration as test_api.py."""
    
    # Set environment variables to match test_api.py configuration
    os.environ["LLM_PROVIDER"] = "openai"  # Use OpenAI provider for proxy compatibility
    os.environ["LLM_MODEL"] = "claude-3-7-sonnet-20250219"  # Same model as test_api.py
    os.environ["LLM_BASE_URL"] = "https://aiproxy.usw.sealos.io/v1"  # Same base URL
    os.environ["LLM_API_KEY"] = "sk-aaoM7y5b82RceVtaMfWy2AkTTvp330MJKRBvCg3c1ysbPEf4"  # Same API key
    
    print("Testing prompt_run with Sealos AI Proxy...")
    print(f"Provider: {os.environ['LLM_PROVIDER']}")
    print(f"Model: {os.environ['LLM_MODEL']}")
    print(f"Base URL: {os.environ['LLM_BASE_URL']}")
    print(f"API Key: {os.environ['LLM_API_KEY'][:20]}...")
    print("-" * 50)
    
    try:
        # Test with the conversation_summary template
        response = await prompt_run(
            "conversation_summary",
            conversation_text="User: What is Sealos?\nAssistant: Sealos is an open-source cloud operating system..."
        )
        
        print("✅ prompt_run successful!")
        print(f"Response type: {type(response)}")
        
        # Try to access the response content
        if hasattr(response, 'content'):
            print(f"Response content: {response.content}")
        elif hasattr(response, 'text'):
            print(f"Response text: {response.text}")
        elif hasattr(response, 'message'):
            print(f"Response message: {response.message}")
        else:
            print(f"Response object: {response}")
            print(f"Response attributes: {dir(response)}")
        
        return response
        
    except Exception as e:
        print(f"❌ prompt_run failed: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_simple_prompt():
    """Test with a simple inline prompt to verify basic functionality."""
    
    print("\n" + "="*50)
    print("Testing with simple inline prompt...")
    print("="*50)
    
    # Set environment variables
    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "claude-3-7-sonnet-20250219"
    os.environ["LLM_BASE_URL"] = "https://aiproxy.usw.sealos.io/v1"
    os.environ["LLM_API_KEY"] = "sk-aaoM7y5b82RceVtaMfWy2AkTTvp330MJKRBvCg3c1ysbPEf4"
    
    try:
        # Import the engine directly for a simpler test
        from codin.prompt.engine import PromptEngine
        from codin.model.factory import create_llm_from_env
        
        # Create LLM instance
        llm = create_llm_from_env()
        print(f"Created LLM: {llm}")
        print(f"LLM type: {type(llm)}")
        
        # Test direct LLM call
        if hasattr(llm, 'generate'):
            result = await llm.generate("What is Sealos? Please provide a brief explanation.")
            print("✅ Direct LLM call successful!")
            print(f"Result: {result}")
        else:
            print(
                f"❌ LLM doesn't have generate method. Available methods: " +
                f"{[m for m in dir(llm) if not m.startswith('_')]}"
            )
        
        return result
        
    except Exception as e:
        print(f"❌ Simple prompt test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main test function."""
    print("Starting prompt_run tests with Sealos AI Proxy configuration...")
    
    # Test 1: Try with existing template
    result1 = await test_prompt_run_with_sealos()
    
    # Test 2: Try simple direct LLM call
    result2 = await test_simple_prompt()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Template test: {'✅ PASSED' if result1 else '❌ FAILED'}")
    print(f"Direct LLM test: {'✅ PASSED' if result2 else '❌ FAILED'}")


if __name__ == "__main__":
    asyncio.run(main()) 
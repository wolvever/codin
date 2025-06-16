#!/usr/bin/env python3
"""
Test script for Sealos AI Proxy API
"""

import requests
import json


def test_sealos_api():
    """Test the Sealos AI Proxy API with the provided parameters."""
    
    # API endpoint
    url = "https://aiproxy.usw.sealos.io/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": "Bearer sk-aaoM7y5b82RceVtaMfWy2AkTTvp330MJKRBvCg3c1ysbPEf4",
        "Content-Type": "application/json"
    }
    
    # Request payload
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "messages": [
            {
                "role": "user",
                "content": "What is Sealos"
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    try:
        print("Making API request to Sealos AI Proxy...")
        print(f"URL: {url}")
        print(f"Model: {data['model']}")
        print(f"Message: {data['messages'][0]['content']}")
        print("-" * 50)
        
        # Make the API request
        response = requests.post(url, headers=headers, json=data)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        print("✅ API request successful!")
        print(f"Status Code: {response.status_code}")
        print("\nResponse:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Extract and display the assistant's response
        if "choices" in result and len(result["choices"]) > 0:
            assistant_message = result["choices"][0]["message"]["content"]
            print("\n" + "="*50)
            print("ASSISTANT RESPONSE:")
            print("="*50)
            print(assistant_message)
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(f"Raw response: {response.text}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


if __name__ == "__main__":
    test_sealos_api() 
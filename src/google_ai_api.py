"""
Google AI API service integration using generativelanguage.googleapis.com endpoint.
"""

import os
import requests
import json
import time
from typing import Optional


def is_available() -> bool:
    """
    Check if Google AI API is available and configured.
    
    Returns:
        bool: True if API key is available, False otherwise
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    return bool(api_key and api_key.strip())


def generate_text(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    max_output_tokens: int = 200,
    temperature: float = 0.3,
    top_p: float = 0.95,
    top_k: int = 40,
    max_retries: int = 3
) -> str:
    """
    Generate text using Google AI API.
    
    Args:
        prompt: The text prompt
        model: Model name (default: gemini-2.5-flash-lite)
        max_output_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_retries: Maximum number of retry attempts
        
    Returns:
        Generated text response
        
    Raises:
        Exception: If API call fails after retries
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("GOOGLE_API_KEY environment variable is not set")
    
    # Google AI API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    # Request headers
    headers = {
        "Content-Type": "application/json"
    }
    
    # Request payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "topK": top_k,
            "topP": top_p,
            "maxOutputTokens": max_output_tokens
        }
    }
    
    # Add API key to URL parameters
    params = {
        "key": api_key
    }
    
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            response = requests.post(
                url, 
                headers=headers, 
                params=params,
                json=payload,
                timeout=30
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                if retry < max_retries - 1:
                    print(f"Rate limited, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise Exception(f"Rate limited after {max_retries} attempts")
            
            # Check for successful response
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract generated text
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if len(parts) > 0 and "text" in parts[0]:
                        return parts[0]["text"]
            
            raise Exception("No valid response content found")
            
        except requests.exceptions.RequestException as e:
            if retry < max_retries - 1:
                print(f"Request failed (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"Request failed after {max_retries} attempts: {e}")
        except Exception as e:
            if retry < max_retries - 1:
                print(f"API call failed (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise Exception(f"API call failed after {max_retries} attempts: {e}")
    
    raise Exception("Unexpected error in generate_text function")


def test_api() -> bool:
    """
    Test the Google AI API with a simple prompt.
    
    Returns:
        bool: True if test successful, False otherwise
    """
    try:
        response = generate_text(
            prompt="Hello, please respond with 'API is working'",
            max_output_tokens=50,
            temperature=0.1
        )
        return "API is working" in response or len(response.strip()) > 0
    except Exception as e:
        print(f"API test failed: {e}")
        return False


if __name__ == "__main__":
    # Load environment variables from .env file when run directly
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Get the project root directory (parent of src/)
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path, override=True)
    
    # Simple test when run directly
    print("Testing Google AI API...")
    print(f"API Key available: {is_available()}")
    print(f"Model choice: {os.getenv('MODEL_CHOICE', 'Not set')}")
    
    if is_available():
        if test_api():
            print("✅ Google AI API is working correctly!")
        else:
            print("❌ Google AI API test failed")
    else:
        print("❌ Google AI API key not configured")
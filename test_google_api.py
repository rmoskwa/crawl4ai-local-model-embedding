#!/usr/bin/env python3
"""
Test script for Google AI API key and model availability.
Tests the Gemini model configuration from the .env file.
"""

import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file."""
    # Load from project root .env file
    project_root = Path(__file__).resolve().parent
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment from: {env_path}")
    else:
        print(f"⚠ No .env file found at: {env_path}")
        return False
    
    return True

def test_google_ai_api():
    """Test Google AI API key and model availability."""
    print("\n" + "="*60)
    print("GOOGLE AI API TEST")
    print("="*60)
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]} (masked)")
    
    # Get model choice
    model_choice = os.getenv("MODEL_CHOICE", "gemini-2.5-flash-lite")
    print(f"✓ Model configured: {model_choice}")
    
    # Test 1: List available models
    print(f"\n{'='*40}")
    print("TEST 1: List Available Models")
    print(f"{'='*40}")
    
    try:
        models_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(models_url, timeout=10)
        
        if response.status_code == 200:
            models_data = response.json()
            models = models_data.get("models", [])
            print(f"✓ API accessible - Found {len(models)} models")
            
            # Check if our specific model is available
            model_names = [model.get("name", "").split("/")[-1] for model in models]
            if model_choice in model_names:
                print(f"✓ Target model '{model_choice}' is available")
                target_available = True
            else:
                print(f"❌ Target model '{model_choice}' NOT found")
                print(f"Available models: {', '.join(model_names[:10])}")
                target_available = False
        else:
            print(f"❌ Models API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing models API: {e}")
        return False
    
    # Test 2: Test content generation
    print(f"\n{'='*40}")
    print("TEST 2: Content Generation Test")
    print(f"{'='*40}")
    
    if not target_available:
        print("⚠ Skipping generation test - target model not available")
        return False
    
    try:
        generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_choice}:generateContent?key={api_key}"
        
        test_payload = {
            "contents": [{
                "parts": [{
                    "text": "Generate a brief summary of this code: print('Hello World')"
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 100
            }
        }
        
        response = requests.post(
            generate_url, 
            json=test_payload, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and result["candidates"]:
                generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✓ Content generation successful!")
                print(f"✓ Response: {generated_text.strip()}")
                return True
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ Content generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg = error_data["error"].get("message", "Unknown error")
                    print(f"Error details: {error_msg}")
            except:
                pass
            
            return False
            
    except Exception as e:
        print(f"❌ Error testing content generation: {e}")
        return False

def test_model_specific_features():
    """Test model-specific features that might cause issues."""
    print(f"\n{'='*40}")
    print("TEST 3: Model Feature Tests")
    print(f"{'='*40}")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    model_choice = os.getenv("MODEL_CHOICE", "gemini-2.5-flash-lite")
    
    # Test with code content (similar to what the app sends)
    test_cases = [
        {
            "name": "Code Summary Test",
            "content": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Usage example
result = calculate_fibonacci(10)
print(f"Fibonacci of 10 is: {result}")
"""
        },
        {
            "name": "MATLAB Code Test", 
            "content": """
function y = makeSincPulse(flip_angle, system, duration)
    % Create a sinc-shaped RF pulse
    t = linspace(-duration/2, duration/2, 1000);
    y = sinc(t * 2) * flip_angle;
end
"""
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nSubtest {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_choice}:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": f"Provide a brief summary of this code:\n\n{test_case['content']}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200
                }
            }
            
            response = requests.post(
                generate_url, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "candidates" in result and result["candidates"]:
                    print(f"✓ {test_case['name']} successful")
                else:
                    print(f"❌ {test_case['name']} - no content generated")
            else:
                print(f"❌ {test_case['name']} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['name']} error: {e}")

def main():
    """Main test function."""
    print("Google AI API Key and Model Availability Test")
    print("=" * 60)
    
    # Load environment
    if not load_environment():
        print("❌ Failed to load environment variables")
        sys.exit(1)
    
    # Test basic API functionality
    if test_google_ai_api():
        print(f"\n{'='*60}")
        print("✓ BASIC API TEST PASSED")
        print("✓ API key is valid and model is accessible")
        
        # Run additional feature tests
        test_model_specific_features()
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS COMPLETED")
        print("Your Google AI API configuration appears to be working!")
        
    else:
        print(f"\n{'='*60}")
        print("❌ API TEST FAILED")
        print("Please check your API key and model configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ollama Chat Example using LiteLLM

This example demonstrates how to use LiteLLM to connect to an Ollama server
running Gemma3 12b model for a simple chat application.

Requirements:
- Ollama server running (default: http://localhost:11434)
- Gemma3:12b model available on the Ollama server

Usage:
    python ollama-example.py
"""

import os
import sys
from typing import List, Dict
import litellm
from litellm import completion
import json
from tools import TOOLS, execute_tool_call, add_task, has_task


def setup_ollama_client(model_name: str):
    """Configure LiteLLM for Ollama
    
    Args:
        model_name: The name of the model to use (default: gemma3:12b)
    
    Returns:
        tuple: (api_base_url, full_model_name)
    """
    # Get API base URL from environment variable or use default
    api_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Set the base URL for Ollama API
    litellm.api_base = api_base
    
    # Create full model name with ollama prefix
    full_model_name = f"ollama/{model_name}"
    
    # Enable debug logging (optional)
    # litellm.set_verbose = True
    
    print("✅ Ollama client configured")
    print(f"📡 API Base URL: {api_base}")
    print(f"🤖 Model: {model_name}")
    
    return api_base, full_model_name


def validate_connection(model_name: str):
    """Test connection to Ollama server
    
    Args:
        model_name: The full model name to test with
    """
    try:
        # Try a simple completion to test connectivity
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=50,
            temperature=0.7
        )
        print("✅ Connection to Ollama server successful!")
        print(f"🤖 Test response: {response.choices[0].message.content}")
        
        # Test tool calling functionality
        print("\n🔧 Testing tool calling functionality...")
        try:
            # Add a validation task using direct function call
            add_task("validate", "Tool calling validation test")
            
            # Test tool calling through the model
            tool_test_response = completion(
                model=model_name,
                messages=[{"role": "user", "content": "Please add a task called 'test' with description 'Tool calling test' and then check if task 'validate' exists."}],
                tools=TOOLS,
                max_tokens=100,
                temperature=0.7
            )
            
            # Check if the model made tool calls
            if tool_test_response.choices[0].message.tool_calls:
                print("✅ Tool calling test successful!")
                
                # Execute any tool calls made by the model
                for tool_call in tool_test_response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    result = execute_tool_call(function_name, arguments)
                    print(f"🔧 Tool call result: {result}")
                
                # Verify the validation task exists
                if has_task("validate"):
                    print("✅ Tool validation successful - task verification passed!")
                else:
                    print("⚠️ Tool validation partial - task verification failed")
                    
            else:
                print("⚠️ Tool calling test failed - model did not make tool calls")
                print("📝 This may indicate the model doesn't support tool calling or needs different prompting")
                
        except Exception as tool_error:
            print(f"⚠️ Tool calling test failed: {tool_error}")
            print("📝 Tool calling validation failed, but basic connection works")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Ollama server: {e}")
        print("\n🔧 Troubleshooting:")
        print(f"1. Ensure Ollama server is running at {litellm.api_base}")
        print("2. Verify that 'gemma3:12b' model is available")
        print("3. Check network connectivity")
        print("4. Set OLLAMA_BASE_URL environment variable if using custom URL")
        return False


def chat_with_ollama(model_name: str = "ollama/gemma3:12b"):
    """Interactive chat with Ollama Gemma model
    
    Args:
        model_name: The full model name to use for chat
    """
    print(f"\n🚀 Starting chat with {model_name}...")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("🔧 Tool calling enabled - you can ask me to manage tasks!")
    print("   Example: 'Add a task called shopping with description buy groceries'")
    print("-" * 50)
    
    # Store conversation history
    conversation_history: List[Dict[str, str]] = []
    
    while True:
        try:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a message or type 'quit' to exit.")
                continue
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            print("🤖 AI: ", end="", flush=True)
            
            # Get response from Ollama with tool calling support
            response = completion(
                model=model_name,
                messages=conversation_history,
                tools=TOOLS,
                max_tokens=500,
                temperature=0.7,
                stream=False  # Set to True for streaming responses
            )
            
            # Check if the model made tool calls
            if response.choices[0].message.tool_calls:
                print("\n🔧 Executing tool calls...")
                
                # Add the assistant's message with tool calls to history (convert to dict)
                assistant_message = response.choices[0].message
                conversation_history.append(assistant_message.model_dump())
                
                # Execute each tool call
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"📞 Calling '{function_name}' with arguments: {arguments}")
                    result = execute_tool_call(function_name, arguments)
                    print(f"📋 Result (Type: {type(result).__name__}) : {result}")

                    # Add tool result to conversation history
                    conversation_history.append({
                        "role": "tool",
                        "content": str(result),  # Ensure result is a string
                        "tool_call_id": str(tool_call.id),  # Ensure tool_call_id is a string
                        "name": function_name
                    })
                
                # Get final response from model with tool results
                final_response = completion(
                    model=model_name,
                    messages=conversation_history,
                    tools=TOOLS,
                    max_tokens=500,
                    temperature=0.7,
                    stream=False
                )
                
                assistant_message = final_response.choices[0].message.content
                print(f"\n🤖 {assistant_message}")
                
                # Add final assistant response to history
                conversation_history.append({"role": "assistant", "content": assistant_message})
                
            else:
                # Regular response without tool calls
                assistant_message = response.choices[0].message.content
                print(assistant_message)
                
                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": assistant_message})
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error during chat: {e}")
            print("Continuing chat... (type 'quit' to exit)")


def main():
    """Main function"""
    print("🌟 Ollama Chat Example with LiteLLM")
    print("=" * 40)

    model_name: str
    model_name = "gemma3:12b"
    #model_name = "qwen3:14b"
    #model_name = "mistral-nemo:12b"
    #model_name = "granite3.2:8b"

    # Setup the client and get configuration
    api_base, full_model_name = setup_ollama_client(model_name)
    
    # Test connection
    if not validate_connection(full_model_name):
        print("\n⚠️  Connection test failed. The chat may not work properly.")
        response = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Exiting...")
            sys.exit(1)
    
    # Start interactive chat
    chat_with_ollama(full_model_name)


if __name__ == "__main__":
    main()
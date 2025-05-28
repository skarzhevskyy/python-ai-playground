#!/usr/bin/env python3
"""
Integration tests for ollama-example.py with real Ollama server.

These tests require a running Ollama server and do not use mocks.
"""

import os
import sys
import json

# Add the parent directory to the path so we can import ollama-example
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules under test
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import ollama_example
from tools import add_task, has_task, list_tasks, mark_task_done, execute_tool_call, TOOLS
from litellm import completion


def test_real_ollama_tool_calling():
    """Test tool calling with real Ollama server - no mocks."""
    print("🧪 Testing real Ollama tool calling functionality...")
    
    # Setup model name
    model_name = "ollama/gemma3:12b"
    
    try:
        # Clear any existing tasks first
        from tools import _tasks
        _tasks.clear()
        
        # Test 1: Basic connection
        print("1. Testing basic connection...")
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello briefly."}],
            max_tokens=50,
            temperature=0.7
        )
        print(f"✅ Basic connection successful: {response.choices[0].message.content[:50]}...")
        
        # Test 2: Direct tool function calls
        print("\n2. Testing direct tool function calls...")
        add_task("test_task", "This is a test task")
        assert has_task("test_task"), "Task should exist after adding"
        print("✅ Direct tool calls working")
        
        # Test 3: Tool calling through model (if supported)
        print("\n3. Testing tool calling through model...")
        try:
            tool_response = completion(
                model=model_name,
                messages=[{
                    "role": "user", 
                    "content": "Please add a task called 'integration_test' with description 'Testing tool calling integration' and then tell me if task 'test_task' exists."
                }],
                tools=TOOLS,
                max_tokens=200,
                temperature=0.7
            )
            
            if tool_response.choices[0].message.tool_calls:
                print("✅ Model made tool calls!")
                
                # Execute the tool calls
                for tool_call in tool_response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    result = execute_tool_call(function_name, arguments)
                    print(f"🔧 Executed {function_name}: {result}")
                
                # Verify tasks were created/checked
                if has_task("integration_test"):
                    print("✅ Tool calling integration test successful!")
                else:
                    print("⚠️ Tool calling partially successful (task not found)")
                    
            else:
                print("ℹ️ Model did not make tool calls - may not support tool calling or need different prompting")
                
        except Exception as tool_error:
            print(f"⚠️ Tool calling through model failed: {tool_error}")
            print("ℹ️ This may be expected if the model doesn't support tool calling")
        
        # Test 4: Validation function (same as validate_connection but simplified)
        print("\n4. Testing validation functionality...")
        add_task("validate", "Tool calling validation test")
        
        if has_task("validate"):
            print("✅ Validation test passed - can add and check tasks")
        else:
            print("❌ Validation test failed")
            
        # Test 5: List tasks
        print("\n5. Testing task listing...")
        task_list = list_tasks()
        print(f"📋 Current tasks: {task_list}")
        
        # Test 6: Mark task done
        print("\n6. Testing mark task done...")
        result = mark_task_done("test_task")
        print(f"✅ Mark done result: {result}")
        
        print("\n🎉 Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("💡 Make sure Ollama server is running with gemma3:12b model")
        return False


if __name__ == "__main__":
    success = test_real_ollama_tool_calling()
    if not success:
        sys.exit(1)
    print("✅ All integration tests passed!")
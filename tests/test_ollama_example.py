#!/usr/bin/env python3
"""
Unit tests for ollama-example.py

Tests the main functions for configuring and using LiteLLM with Ollama.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

# Add the parent directory to the path so we can import ollama-example
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module under test by renaming the file
import shutil
import tempfile

# Create a temporary module file with proper name for importing
temp_dir = tempfile.mkdtemp()
temp_module_path = os.path.join(temp_dir, "ollama_example.py")
original_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "ollama-example.py")
shutil.copy2(original_path, temp_module_path)
sys.path.insert(0, temp_dir)

import ollama_example


class TestSetupOllamaClient:
    """Test the setup_ollama_client function"""
    
    def test_setup_with_default_values(self):
        """Test setup with default model and URL"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('ollama_example.litellm') as mock_litellm:
                with patch('builtins.print') as mock_print:
                    api_base, full_model_name = ollama_example.setup_ollama_client()
                    
                    # Check that the correct values are set
                    assert api_base == "http://localhost:11434"
                    assert full_model_name == "ollama/gemma3:12b"
                    assert mock_litellm.api_base == "http://localhost:11434"
                    
                    # Check that the correct messages are printed
                    mock_print.assert_any_call("✅ Ollama client configured")
                    mock_print.assert_any_call("📡 API Base URL: http://localhost:11434")
                    mock_print.assert_any_call("🤖 Model: gemma3:12b")
    
    def test_setup_with_custom_model(self):
        """Test setup with custom model name"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('ollama_example.litellm') as mock_litellm:
                with patch('builtins.print') as mock_print:
                    api_base, full_model_name = ollama_example.setup_ollama_client("custom-model:7b")
                    
                    assert api_base == "http://localhost:11434"
                    assert full_model_name == "ollama/custom-model:7b"
                    assert mock_litellm.api_base == "http://localhost:11434"
                    
                    mock_print.assert_any_call("🤖 Model: custom-model:7b")
    
    def test_setup_with_environment_variable(self):
        """Test setup with OLLAMA_BASE_URL environment variable"""
        custom_url = "http://custom-server:8080"
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": custom_url}):
            with patch('ollama_example.litellm') as mock_litellm:
                with patch('builtins.print') as mock_print:
                    api_base, full_model_name = ollama_example.setup_ollama_client()
                    
                    assert api_base == custom_url
                    assert full_model_name == "ollama/gemma3:12b"
                    assert mock_litellm.api_base == custom_url
                    
                    mock_print.assert_any_call(f"📡 API Base URL: {custom_url}")


class TestTestConnection:
    """Test the validate_connection function"""
    
    def test_successful_connection(self):
        """Test successful connection to Ollama server"""
        # Mock a successful completion response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello! How can I help you today?"
        
        with patch('ollama_example.completion', return_value=mock_response) as mock_completion:
            with patch('builtins.print') as mock_print:
                result = ollama_example.validate_connection("ollama/gemma3:12b")
                
                assert result is True
                mock_completion.assert_called_once_with(
                    model="ollama/gemma3:12b",
                    messages=[{"role": "user", "content": "Say hello!"}],
                    max_tokens=50,
                    temperature=0.7
                )
                
                mock_print.assert_any_call("✅ Connection to Ollama server successful!")
                mock_print.assert_any_call("🤖 Test response: Hello! How can I help you today?")
    
    def test_failed_connection(self):
        """Test failed connection to Ollama server"""
        with patch('ollama_example.completion', side_effect=Exception("Connection refused")) as mock_completion:
            with patch('ollama_example.litellm') as mock_litellm:
                mock_litellm.api_base = "http://localhost:11434"
                with patch('builtins.print') as mock_print:
                    result = ollama_example.validate_connection("ollama/gemma3:12b")
                    
                    assert result is False
                    mock_print.assert_any_call("❌ Failed to connect to Ollama server: Connection refused")
                    mock_print.assert_any_call("\n🔧 Troubleshooting:")
                    mock_print.assert_any_call("1. Ensure Ollama server is running at http://localhost:11434")
                    mock_print.assert_any_call("2. Verify that 'gemma3:12b' model is available")
    
    def test_default_model_parameter(self):
        """Test that validate_connection uses default model when none provided"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello!"
        
        with patch('ollama_example.completion', return_value=mock_response) as mock_completion:
            with patch('builtins.print'):
                ollama_example.validate_connection()
                
                mock_completion.assert_called_once_with(
                    model="ollama/gemma3:12b",
                    messages=[{"role": "user", "content": "Say hello!"}],
                    max_tokens=50,
                    temperature=0.7
                )


class TestChatWithOllama:
    """Test the chat_with_ollama function"""
    
    def test_chat_quit_command(self):
        """Test that chat exits properly on quit command"""
        with patch('builtins.input', side_effect=['quit']):
            with patch('builtins.print') as mock_print:
                ollama_example.chat_with_ollama("ollama/gemma3:12b")
                
                mock_print.assert_any_call("\n🚀 Starting chat with ollama/gemma3:12b...")
                mock_print.assert_any_call("\n👋 Goodbye!")
    
    def test_chat_exit_command(self):
        """Test that chat exits properly on exit command"""
        with patch('builtins.input', side_effect=['exit']):
            with patch('builtins.print') as mock_print:
                ollama_example.chat_with_ollama("ollama/gemma3:12b")
                
                mock_print.assert_any_call("\n👋 Goodbye!")
    
    def test_chat_empty_input(self):
        """Test that chat handles empty input correctly"""
        with patch('builtins.input', side_effect=['', 'quit']):
            with patch('builtins.print') as mock_print:
                ollama_example.chat_with_ollama("ollama/gemma3:12b")
                
                mock_print.assert_any_call("Please enter a message or type 'quit' to exit.")
    
    def test_chat_with_response(self):
        """Test normal chat interaction with a response"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test response"
        
        with patch('builtins.input', side_effect=['Hello', 'quit']):
            with patch('ollama_example.completion', return_value=mock_response) as mock_completion:
                with patch('builtins.print') as mock_print:
                    ollama_example.chat_with_ollama("ollama/test-model")
                    
                    # Check that completion was called exactly once
                    assert mock_completion.call_count == 1
                    
                    # Check the call parameters 
                    call_args = mock_completion.call_args
                    assert call_args.kwargs['model'] == "ollama/test-model"
                    # The messages should only contain the user message at the time of the call
                    messages = call_args.kwargs['messages']
                    assert len(messages) >= 1
                    assert messages[0] == {"role": "user", "content": "Hello"}
                    assert call_args.kwargs['max_tokens'] == 500
                    assert call_args.kwargs['temperature'] == 0.7
                    assert call_args.kwargs['stream'] is False
                    
                    # Check that response was printed
                    mock_print.assert_any_call("This is a test response")
    
    def test_chat_api_error(self):
        """Test chat handling of API errors"""
        with patch('builtins.input', side_effect=['Hello', 'quit']):
            with patch('ollama_example.completion', side_effect=Exception("API Error")):
                with patch('builtins.print') as mock_print:
                    ollama_example.chat_with_ollama("ollama/gemma3:12b")
                    
                    mock_print.assert_any_call("\n❌ Error during chat: API Error")
                    mock_print.assert_any_call("Continuing chat... (type 'quit' to exit)")
    
    def test_chat_keyboard_interrupt(self):
        """Test chat handling of keyboard interrupt"""
        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            with patch('builtins.print') as mock_print:
                ollama_example.chat_with_ollama("ollama/gemma3:12b")
                
                mock_print.assert_any_call("\n\n👋 Chat interrupted. Goodbye!")
    
    def test_default_model_parameter_chat(self):
        """Test that chat_with_ollama uses default model when none provided"""
        with patch('builtins.input', side_effect=['quit']):
            with patch('builtins.print') as mock_print:
                ollama_example.chat_with_ollama()
                
                mock_print.assert_any_call("\n🚀 Starting chat with ollama/gemma3:12b...")


class TestMainFunction:
    """Test the main function"""
    
    def test_main_successful_flow(self):
        """Test main function with successful connection"""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello!"
        
        with patch('ollama_example.setup_ollama_client', return_value=("http://localhost:11434", "ollama/gemma3:12b")):
            with patch('ollama_example.validate_connection', return_value=True):
                with patch('ollama_example.chat_with_ollama') as mock_chat:
                    with patch('builtins.print'):
                        ollama_example.main()
                        
                        mock_chat.assert_called_once_with("ollama/gemma3:12b")
    
    def test_main_failed_connection_continue(self):
        """Test main function with failed connection but user chooses to continue"""
        with patch('ollama_example.setup_ollama_client', return_value=("http://localhost:11434", "ollama/gemma3:12b")):
            with patch('ollama_example.validate_connection', return_value=False):
                with patch('builtins.input', return_value='y'):
                    with patch('ollama_example.chat_with_ollama') as mock_chat:
                        with patch('builtins.print'):
                            ollama_example.main()
                            
                            mock_chat.assert_called_once_with("ollama/gemma3:12b")
    
    def test_main_failed_connection_exit(self):
        """Test main function with failed connection and user chooses to exit"""
        with patch('ollama_example.setup_ollama_client', return_value=("http://localhost:11434", "ollama/gemma3:12b")):
            with patch('ollama_example.validate_connection', return_value=False):
                with patch('builtins.input', return_value='n'):
                    with patch('ollama_example.chat_with_ollama') as mock_chat:
                        with patch('builtins.print'):
                            with pytest.raises(SystemExit) as exc_info:
                                ollama_example.main()
                            
                            assert exc_info.value.code == 1
                            mock_chat.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
# Python AI Playground

## Setup Instructions

1.  **Install uv:**
    
    If you don't have `uv` installed, you can install it by following the official instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).
    
    For example, on macOS and Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Create a virtual environment:**
    
    Navigate to the project directory and create a virtual environment using `uv`:
    ```bash
    uv venv
    ```
    This will create a `.venv` directory in your project.

3.  **Activate the virtual environment:**
    
    On macOS and Linux:
    ```bash
    source .venv/bin/activate
    ```
    On Windows (Git Bash or WSL):
    ```bash
    source .venv/Scripts/activate
    ```
    On Windows (Command Prompt):
    ```bash
    .venv\Scripts\activate.bat
    ```
    On Windows (PowerShell):
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

4.  **Install dependencies:**
    
    Once the virtual environment is activated, install the project dependencies:
    
    **Option 1: Using uv (if available):**
    ```bash
    uv pip sync pyproject.toml
    ```
    Alternatively, for an editable install:
    ```bash
    uv pip install -e .
    ```
    
    **Option 2: Using pip:**
    ```bash
    pip install -e .
    ```
    Or install dependencies directly:
    ```bash
    pip install litellm
    ```

Now you're ready to start working on the project!

## Examples

### Ollama Chat Example

The `src/ollama-example.py` file demonstrates how to use LiteLLM to connect to an Ollama server running Gemma3 12b model for a chat application.

**Prerequisites:**
- Ollama server running (default: `http://localhost:11434`)
- Gemma3:12b model available on the Ollama server

**Configuration:**
- **OLLAMA_BASE_URL**: Environment variable to set the Ollama server URL (default: `http://localhost:11434`)
- **Model**: Configurable model name (default: `gemma3:12b`)

**To run the example:**

```bash
# Using default settings
python src/ollama-example.py

# Using custom Ollama server URL
OLLAMA_BASE_URL=http://your-server:11434 python src/ollama-example.py
```

**Features:**
- Interactive chat interface with Gemma3 12b model
- **Tool calling functionality with task management**
- Configurable model name and server URL
- Connection testing to Ollama server
- Environment variable support for flexible deployment
- Conversation history management
- Error handling and graceful fallbacks
- Clear troubleshooting instructions

**Tool Calling Features:**
The example now includes tool calling capabilities that allow the AI model to:
- Add tasks with names and descriptions
- List all current tasks
- Mark tasks as completed
- Check if specific tasks exist

You can interact with these tools by asking the AI to manage tasks, for example:
- "Add a task called 'shopping' with description 'buy groceries'"
- "List all my tasks"
- "Mark the shopping task as done"
- "Do I have a task called 'homework'?"

**Running from PyCharm:**
1. Open the project in PyCharm
2. Ensure the virtual environment is configured as the project interpreter
3. Set environment variables if needed in Run Configuration
4. Right-click on `src/ollama-example.py` and select "Run 'ollama-example'"
5. Or use the green play button when the file is open

The example will automatically test the connection to the Ollama server and provide helpful error messages if the connection fails.

## Testing

Run the unit tests to verify functionality:

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_ollama_example.py -v

# Run integration test with real Ollama (requires running server)
python tests/test_ollama_integration.py
```

**Test Coverage:**
- Configuration setup with environment variables
- Connection testing
- Chat functionality
- Tool calling functionality
- Task management
- Error handling
- Integration test with real Ollama server
- Model name configuration


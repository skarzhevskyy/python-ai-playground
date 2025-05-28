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
    
    Once the virtual environment is activated, install the project dependencies listed in your `pyproject.toml` file:
    ```bash
    uv pip sync pyproject.toml
    ```
    Alternatively, for an editable install, you can use:
    ```bash
    uv pip install -e .
    ```
    Add your project's dependencies to the `[project.dependencies]` section in the `pyproject.toml` file.

Now you're ready to start working on the project!


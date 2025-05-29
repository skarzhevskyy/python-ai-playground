#!/usr/bin/env python3
"""
Task management tools for the Ollama example.

This module provides simple task management functionality that can be called
by AI models through tool calling.
"""

from typing import List, Dict, Any
import json


# In-memory task storage (in a real app, this would be a database)
_tasks: Dict[str, Dict[str, Any]] = {}


def add_task(name: str, description: str) -> str:
    """Add a new task.
    
    Args:
        name: The name/identifier of the task
        description: A description of what the task involves
        
    Returns:
        Success message
    """
    print(f"    🐞 - add task '{name}'")
    _tasks[name] = {
        "name": name,
        "description": description,
        "completed": False
    }
    return f"Task '{name}' added successfully. Task description is: {description}"


def list_tasks() -> str:
    """List all tasks.
    
    Returns:
        A formatted string of all tasks
    """
    if not _tasks:
        return "No tasks found."
    
    task_list = []
    for task in _tasks.values():
        status = "✅ Done" if task["completed"] else "⏳ Pending"
        task_list.append(f"- {task['name']}: {task['description']} ({status})")
    
    return "Current tasks:\n" + "\n".join(task_list)


def mark_task_done(name: str) -> str:
    """Mark a task as completed.
    
    Args:
        name: The name of the task to mark as done
        
    Returns:
        Success or error message
    """
    if name not in _tasks:
        return f"Task '{name}' not found."
    
    _tasks[name]["completed"] = True
    return f"Task '{name}' marked as completed."


def has_task(name: str) -> bool:
    """Check if a task exists.
    
    Args:
        name: The name of the task to check
        
    Returns:
        True if task exists, False otherwise
    """
    task_exists : bool = name in _tasks
    print(f"    🐞 - task '{name}' exists {task_exists}")
    return task_exists


# Tool definitions for LiteLLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Add a new task with a name and description",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name/identifier of the task"
                    },
                    "description": {
                        "type": "string", 
                        "description": "A description of what the task involves"
                    }
                },
                "required": ["name", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_tasks",
            "description": "List all current tasks",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "mark_task_done",
            "description": "Mark a task as completed",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the task to mark as done"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "has_task",
            "description": "Check if a task exists",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the task to check"
                    }
                },
                "required": ["name"]
            }
        }
    }
]


def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool call by name with given arguments.
    
    Args:
        tool_name: The name of the tool function to call
        arguments: Dictionary of arguments to pass to the function
        
    Returns:
        The result of the tool call as a string
    """
    # Map tool names to functions
    tool_functions = {
        "add_task": add_task,
        "list_tasks": list_tasks,
        "mark_task_done": mark_task_done,
        "has_task": has_task
    }
    
    if tool_name not in tool_functions:
        return f"Unknown tool: {tool_name}"
    
    try:
        func = tool_functions[tool_name]
        result = func(**arguments)
        # Convert boolean results to string for consistency
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
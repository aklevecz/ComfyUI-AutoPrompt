# __init__.py
"""
Ollama ComfyUI Nodes Package

This package provides ComfyUI nodes for integrating with Ollama locally:
- OllamaPromptGenerator: Generate prompts and text using Ollama models
- OllamaModelLister: List available Ollama models
- OllamaChat: Chat with Ollama models with conversation history

Installation:
1. Place this folder in your ComfyUI/custom_nodes/ directory
2. Install requirements: pip install requests
3. Make sure Ollama is running locally (ollama serve)
4. Restart ComfyUI

Usage:
- Download models with: ollama pull llama3.2
- Find nodes under "AI/Ollama" category in ComfyUI
"""

from .ollama_nodes import (
    OllamaPromptGenerator,
    OllamaModelLister, 
    OllamaChat
)

from .text_viewer import TextDisplay

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "OllamaPromptGenerator": OllamaPromptGenerator,
    "OllamaModelLister": OllamaModelLister,
    "OllamaChat": OllamaChat,
    "TextDisplay": TextDisplay
}

# Display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaPromptGenerator": "Ollama Prompt Generator",
    "OllamaModelLister": "Ollama Model Lister", 
    "OllamaChat": "Ollama Chat",
    "TextDisplay": "Text Display"
}

# Package metadata
__version__ = "1.0.0"
__author__ = "ComfyUI Community"
__description__ = "Ollama integration nodes for ComfyUI"

# Export all required symbols
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "OllamaPromptGenerator",
    "OllamaModelLister",
    "OllamaChat",
]
# ollama_nodes.py
import requests
import json
from typing import Dict, Any, Tuple, List, Optional

class OllamaPromptGenerator:
    """
    ComfyUI Node for generating prompts using Ollama locally
    """
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {
                    "default": "llama3.2",
                    "placeholder": "Model name (e.g., llama3.2, mistral, codellama)"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate a creative prompt for image generation:",
                    "placeholder": "Enter your prompt or instruction"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful AI assistant that generates creative and detailed prompts for image generation.",
                    "placeholder": "System prompt to guide the model's behavior"
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Controls randomness in generation"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum number of tokens to generate"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "tooltip": "Nucleus sampling parameter"
                }),
                "top_k": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Top-k sampling parameter"
                }),
                "host": ("STRING", {
                    "default": "localhost:11434",
                    "placeholder": "Ollama host:port"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "step": 1,
                    "tooltip": "Random seed for reproducibility"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("generated_prompt", "model_info", "raw_response")
    FUNCTION = "generate_prompt"
    CATEGORY = "AI/Ollama"
    
    def check_ollama_connection(self, host: str) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            base_url = f"http://{host}" if not host.startswith('http') else host
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_models(self, host: str) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            base_url = f"http://{host}" if not host.startswith('http') else host
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except Exception:
            pass
        return []
    
    def generate_with_ollama(self, host: str, model: str, prompt: str, 
                            system_prompt: str, temperature: float, 
                            max_tokens: int, top_p: float, top_k: int) -> Dict[str, Any]:
        """Generate text using Ollama API"""
        base_url = f"http://{host}" if not host.startswith('http') else host
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "top_k": top_k
            }
        }
        
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timed out. The model might be loading or the prompt is too complex.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {base_url}. Make sure Ollama is running.")
        except Exception as e:
            raise RuntimeError(f"Error generating with Ollama: {str(e)}")
    
    def generate_prompt(self, model: str, prompt: str, system_prompt: str,
                       temperature: float = 0.7, max_tokens: int = 512,
                       top_p: float = 0.9, top_k: int = 40,
                       seed: int = 0,
                       host: str = "localhost:11434") -> Tuple[str, str, str]:
        """Main execution function"""
        
        # Add automatic randomness to prevent caching
        import random
        import time
        cache_breaker = f"__{random.randint(0, 999999)}_{int(time.time())}"
        seed = random.randint(0, 999999)  # Random seed for reproducibility
        print(f"Cache breaker: {cache_breaker}, Seed: {seed}")
        # Check connection
        if not self.check_ollama_connection(host):
            error_msg = f"Cannot connect to Ollama at {host}. Please ensure Ollama is running."
            return (error_msg, "Connection Error", json.dumps({"error": error_msg}))
        
        # Get available models
        available_models = self.get_available_models(host)
        
        try:
            # Generate text (cache_breaker ensures this always runs fresh)
            print(f"Generating with model: {model} (cache breaker: {cache_breaker})")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            
            response = self.generate_with_ollama(
                host, model, prompt, system_prompt,
                temperature, max_tokens, top_p, top_k
            )
            
            # Extract generated text
            generated_text = response.get('response', '').strip()
            
            # Prepare model info
            model_info = {
                "model": model,
                "available_models": available_models,
                "host": host,
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": top_k
                },
                "generated_at": time.time()
            }
            
            print(f"Generated {len(generated_text)} characters")
            
            return (
                generated_text,
                json.dumps(model_info, indent=2),
                json.dumps(response, indent=2)
            )
            
        except Exception as e:
            error_msg = f"Ollama generation failed: {str(e)}"
            print(f"Error: {error_msg}")
            
            return (
                f"Error: {error_msg}",
                json.dumps({"error": error_msg, "available_models": available_models}),
                json.dumps({"error": error_msg})
            )


class OllamaModelLister:
    """
    ComfyUI Node for listing available Ollama models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "host": ("STRING", {
                    "default": "localhost:11434",
                    "placeholder": "Ollama host:port"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("models_list", "connection_status")
    FUNCTION = "list_models"
    CATEGORY = "AI/Ollama"
    
    def list_models(self, host: str) -> Tuple[str, str]:
        """List all available models"""
        try:
            base_url = f"http://{host}" if not host.startswith('http') else host
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                model_list = []
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    modified = model.get('modified_at', 'Unknown')
                    
                    # Format size
                    if size > 0:
                        if size > 1024**3:
                            size_str = f"{size / (1024**3):.1f} GB"
                        elif size > 1024**2:
                            size_str = f"{size / (1024**2):.1f} MB"
                        else:
                            size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = "Unknown size"
                    
                    model_list.append(f"• {name} ({size_str})")
                
                models_text = "\n".join(model_list) if model_list else "No models found"
                status = f"Connected to {host} - Found {len(models)} models"
                
                return (models_text, status)
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return (f"Error: {error_msg}", f"Connection failed: {error_msg}")
                
        except Exception as e:
            error_msg = f"Cannot connect to Ollama: {str(e)}"
            return (f"Error: {error_msg}", f"Connection failed: {error_msg}")


class OllamaChat:
    """
    ComfyUI Node for chat conversations with Ollama
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {
                    "default": "llama3.2",
                    "placeholder": "Model name"
                }),
                "message": ("STRING", {
                    "multiline": True,
                    "default": "Hello! How can you help me today?",
                    "placeholder": "Your message"
                }),
            },
            "optional": {
                "conversation_history": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Previous conversation (JSON format)"
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "placeholder": "System prompt"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "host": ("STRING", {
                    "default": "localhost:11434",
                    "placeholder": "Ollama host:port"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "updated_history")
    FUNCTION = "chat"
    CATEGORY = "AI/Ollama"
    
    def chat(self, model: str, message: str, conversation_history: str = "",
             system_prompt: str = "You are a helpful assistant.",
             temperature: float = 0.7, host: str = "localhost:11434") -> Tuple[str, str]:
        """Chat with Ollama model"""
        
        # Add automatic randomness to prevent caching
        import random
        import time
        cache_breaker = f"__{random.randint(0, 999999)}_{int(time.time())}"
        
        try:
            base_url = f"http://{host}" if not host.startswith('http') else host
            
            print(f"Chat with {model} (cache breaker: {cache_breaker})")
            
            # Parse conversation history
            messages = []
            if conversation_history.strip():
                try:
                    messages = json.loads(conversation_history)
                except json.JSONDecodeError:
                    messages = []
            
            # Add system message if not present
            if not messages or messages[0].get('role') != 'system':
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": message})
            
            # Prepare payload
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_message = data.get('message', {}).get('content', '')
                
                # Add assistant response to history
                messages.append({"role": "assistant", "content": assistant_message})
                
                return (
                    assistant_message,
                    json.dumps(messages, indent=2)
                )
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return (f"Error: {error_msg}", conversation_history)
                
        except Exception as e:
            error_msg = f"Chat failed: {str(e)}"
            return (f"Error: {error_msg}", conversation_history)
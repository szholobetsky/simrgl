"""
LLM Integration for RAG Pipeline
Supports local models (Qwen, etc.) and API-based models (OpenAI-compatible)
"""

from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str  # 'local', 'openai', 'ollama', 'lmstudio'
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    api_base: Optional[str] = None
    api_key: Optional[str] = None


class BaseLLM(ABC):
    """Base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response from LLM"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available"""
        pass


class LocalLLM(BaseLLM):
    """Local LLM using transformers (Qwen, etc.)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(self):
        """Load the model (lazy loading)"""
        if self.model is not None:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading {self.config.model_name} on {self.device}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response"""
        self.load_model()

        try:
            import torch

            # Format prompt with system message if provided
            if system_prompt:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                full_prompt = prompt

            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            response = response.replace(full_prompt, "").strip()

            return response
        except Exception as e:
            return f"Error generating response: {e}"

    def is_available(self) -> bool:
        """Check if transformers and torch are available"""
        try:
            import torch
            import transformers
            return True
        except ImportError:
            return False


class OpenAILLM(BaseLLM):
    """OpenAI-compatible API (OpenAI, LMStudio, Ollama with openai wrapper)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None

    def _init_client(self):
        """Initialize OpenAI client"""
        if self.client is not None:
            return

        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.api_base
            )
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using OpenAI API"""
        self._init_client()

        try:
            messages = []

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )

            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling API: {e}"

    def is_available(self) -> bool:
        """Check if openai package is available"""
        try:
            import openai
            return True
        except ImportError:
            return False


class OllamaLLM(BaseLLM):
    """Ollama local LLM server"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_base:
            config.api_base = "http://localhost:11434"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using Ollama API"""
        try:
            import requests

            # Use /api/chat endpoint (new Ollama API)
            url = f"{self.config.api_base}/api/chat"

            # Build messages array for chat API
            messages = []
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            messages.append({
                "role": "user",
                "content": prompt
            })

            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p
                }
            }

            # Increase timeout for CPU-based processing (10 minutes)
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()

            result = response.json()
            # Extract response from message content
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            return result.get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {e}"

    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            import requests
            response = requests.get(f"{self.config.api_base}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False


class LLMFactory:
    """Factory for creating LLM instances"""

    @staticmethod
    def create(config: LLMConfig) -> BaseLLM:
        """Create LLM instance based on provider"""
        if config.provider == "local":
            return LocalLLM(config)
        elif config.provider in ["openai", "lmstudio"]:
            return OpenAILLM(config)
        elif config.provider == "ollama":
            return OllamaLLM(config)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")


class RAGWithLLM:
    """RAG pipeline with LLM for generating recommendations"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def create_system_prompt(self) -> str:
        """Create system prompt for code navigation assistant"""
        return """You are an expert software development assistant specializing in code navigation and architecture analysis.

Your role is to:
1. Analyze the provided code context and historical information
2. Identify the most relevant modules and files for the given task
3. Provide specific, actionable recommendations
4. Highlight potential concerns or pitfalls
5. Suggest the best starting points for implementation

Be concise, technical, and focus on actionable insights."""

    def create_user_prompt(self, augmented_context: str) -> str:
        """Create user prompt from augmented context"""
        return f"""{augmented_context}

Based on the above context, please provide:

1. **Most Relevant Locations**: Which modules/files are most relevant and why?
2. **Recommended Approach**: Where should the developer start? What's the suggested implementation path?
3. **Similar Patterns**: What can be learned from the similar historical tasks?
4. **Potential Concerns**: Are there any risks, edge cases, or architectural considerations?
5. **Code Insights**: Any specific observations from the code snippets provided?

Provide specific, actionable recommendations."""

    def generate_recommendations(self, augmented_context: str) -> str:
        """
        Generate LLM recommendations based on augmented context

        Args:
            augmented_context: The augmented prompt from RAG pipeline

        Returns:
            LLM-generated recommendations
        """
        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(augmented_context)

        response = self.llm.generate(user_prompt, system_prompt)
        return response


# Predefined LLM configurations
PREDEFINED_LLMS = {
    "qwen-2.5-coder-1.5b": LLMConfig(
        provider="ollama",  # Changed from "local" to use Ollama
        model_name="qwen2.5-coder:1.5b",  # Ollama model name
        temperature=0.7,
        max_tokens=2000
    ),
    "qwen-2.5-coder-7b": LLMConfig(
        provider="ollama",  # Changed from "local" to use Ollama
        model_name="qwen2.5-coder:7b",  # Ollama model name
        temperature=0.7,
        max_tokens=2000
    ),
    "ollama-qwen": LLMConfig(
        provider="ollama",
        model_name="qwen2.5-coder:latest",
        temperature=0.7,
        max_tokens=2000
    ),
    "ollama-codellama": LLMConfig(
        provider="ollama",
        model_name="codellama:latest",
        temperature=0.7,
        max_tokens=2000
    ),
    "lmstudio": LLMConfig(
        provider="lmstudio",
        model_name="local-model",  # LMStudio uses currently loaded model
        api_base="http://localhost:1234/v1",
        temperature=0.7,
        max_tokens=2000
    ),
    "openai-gpt4": LLMConfig(
        provider="openai",
        model_name="gpt-4",
        api_base="https://api.openai.com/v1",
        temperature=0.7,
        max_tokens=2000
    ),
}


if __name__ == "__main__":
    # Test LLM integration
    print("Testing LLM Integration...")

    # Test Ollama (if available)
    ollama_config = PREDEFINED_LLMS["ollama-qwen"]
    ollama_llm = LLMFactory.create(ollama_config)

    if ollama_llm.is_available():
        print("\n✓ Ollama is available")
        test_prompt = "Explain what a memory leak is in one sentence."
        print(f"\nTest prompt: {test_prompt}")
        response = ollama_llm.generate(test_prompt)
        print(f"Response: {response}")
    else:
        print("\n✗ Ollama is not available")
        print("  Install: https://ollama.ai/")
        print("  Run: ollama pull qwen2.5-coder")

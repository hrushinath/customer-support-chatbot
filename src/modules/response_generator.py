"""
Response Generator Module
Generates responses using local LLM (via Ollama) with RAG context
"""

import logging
import json
from typing import Dict, Optional
from urllib.error import URLError
import urllib.request

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates customer support responses using a local LLM.
    
    Uses:
    - Retrieved context from vector DB (RAG)
    - System prompt to guide behavior
    - Local Ollama for inference
    
    Result: Grounded, hallucination-free responses
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "mistral",
        system_prompt: str = None,
        query_template: str = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 500,
        timeout: int = 120
    ):
        """
        Initialize response generator
        
        Args:
            ollama_url: Base URL for Ollama API
            model_name: Model to use (must be pulled with `ollama pull <model>`)
            system_prompt: System message to guide LLM
            query_template: Template for combining context + question
            temperature: 0.0-1.0 (0=deterministic, 1=random)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum response length
            timeout: Request timeout in seconds
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Set prompts
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.query_template = query_template or self._default_query_template()
        
        # Test connection to Ollama
        self._test_connection()
        
        logger.info(
            f"ResponseGenerator initialized: "
            f"model={model_name}, url={ollama_url}"
        )

    @staticmethod
    def _default_system_prompt() -> str:
        """Default system prompt for customer support"""
        return """You are a helpful and professional customer support assistant.
Your role is to answer customer questions accurately and courteously.

IMPORTANT RULES:
1. Answer ONLY using the provided context below
2. If the answer is not in the context, say: "I don't have information about that"
3. Be concise and clear in your responses
4. If you're unsure, express appropriate uncertainty
5. Always be respectful and professional

Do NOT:
- Make up information
- Provide generic responses not supported by context
- Pretend to know things outside the provided context"""

    @staticmethod
    def _default_query_template() -> str:
        """Default template for combining context and query"""
        return """Context information:
{context}

User Question: {question}

Please answer the question based ONLY on the context provided above."""

    def _test_connection(self) -> bool:
        """
        Test connection to Ollama
        
        Returns:
            True if connection successful
            
        Raises:
            ConnectionError if cannot reach Ollama
        """
        try:
            url = f"{self.ollama_url}/api/tags"
            request = urllib.request.Request(url, method='GET')
            
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    logger.info("Connected to Ollama API")
                    return True
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            raise ConnectionError(
                f"Cannot reach Ollama at {self.ollama_url}\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {str(e)}"
            )

    def generate_response(
        self,
        question: str,
        context: str,
        include_reasoning: bool = False
    ) -> Dict:
        """
        Generate a response to the user question using context
        
        Args:
            question: User's question
            context: Retrieved context from vector DB
            include_reasoning: Include thinking process
            
        Returns:
            Dict with:
            - 'answer': The generated response
            - 'raw_response': Full LLM output
            - 'reasoning': Optional reasoning
            - 'success': Whether generation succeeded
            - 'error': Error message if failed
        """
        try:
            # Build the prompt
            prompt = self.query_template.format(
                context=context,
                question=question
            )
            
            logger.debug(f"Prompt length: {len(prompt)} chars")
            
            # Call Ollama
            response = self._call_ollama(prompt)
            
            if not response['success']:
                return response
            
            # Extract answer
            answer = response['response'].strip()
            
            return {
                'success': True,
                'answer': answer,
                'raw_response': response['response'],
                'reasoning': None,
                'model': self.model_name,
                'tokens_used': response.get('tokens_used', -1)
            }
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'success': False,
                'answer': None,
                'raw_response': None,
                'reasoning': None,
                'error': str(e)
            }

    def _call_ollama(self, prompt: str) -> Dict:
        """
        Call Ollama API to generate text
        
        Args:
            prompt: The complete prompt to send to LLM
            
        Returns:
            Dict with response and metadata
        """
        try:
            url = f"{self.ollama_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            
            # Make request
            data = json.dumps(payload).encode('utf-8')
            request = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            logger.debug(f"Calling Ollama: {self.model_name}")
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode('utf-8'))
            
            if 'response' not in response_data:
                return {
                    'success': False,
                    'response': None,
                    'error': 'No response from model'
                }
            
            return {
                'success': True,
                'response': response_data['response'],
                'tokens_used': response_data.get('tokens_used', -1)
            }
        
        except urllib.error.HTTPError as e:
            logger.error(f"HTTP error from Ollama: {e.code} {e.reason}")
            if e.code == 404:
                return {
                    'success': False,
                    'response': None,
                    'error': f"Model '{self.model_name}' not found. "
                            f"Pull it first: ollama pull {self.model_name}"
                }
            else:
                return {
                    'success': False,
                    'response': None,
                    'error': f"HTTP {e.code}: {e.reason}"
                }
        
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return {
                'success': False,
                'response': None,
                'error': str(e)
            }

    def answer_query(
        self,
        question: str,
        context: str,
        context_info: Dict = None
    ) -> Dict:
        """
        Generate a complete answer with metadata
        
        Args:
            question: User question
            context: Retrieved context
            context_info: Additional context info (sources, confidence, etc.)
            
        Returns:
            Dict with complete answer information
        """
        if not context or len(context.strip()) == 0:
            return {
                'question': question,
                'answer': "I don't have information about that in my knowledge base.",
                'confidence': 'low',
                'sources': [],
                'success': True,
                'no_context': True
            }
        
        # Generate response
        result = self.generate_response(question, context)
        
        if not result['success']:
            return {
                'question': question,
                'answer': f"I encountered an error: {result.get('error', 'Unknown error')}",
                'confidence': 'low',
                'sources': [],
                'success': False,
                'error': result.get('error')
            }
        
        # Build final response
        answer_dict = {
            'question': question,
            'answer': result['answer'],
            'confidence': context_info.get('confidence', 'medium') if context_info else 'medium',
            'sources': context_info.get('sources', []) if context_info else [],
            'success': True,
            'no_context': False,
            'context_chunks': context_info.get('chunks_count', 0) if context_info else 0,
            'avg_similarity': context_info.get('avg_similarity', 0) if context_info else 0
        }
        
        return answer_dict

    def set_parameters(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Update LLM parameters
        
        Args:
            temperature: Generation randomness
            top_p: Nucleus sampling parameter
            max_tokens: Max response length
        """
        if temperature is not None:
            if not (0.0 <= temperature <= 1.0):
                raise ValueError("Temperature must be between 0.0 and 1.0")
            self.temperature = temperature
            logger.info(f"Updated temperature to {temperature}")
        
        if top_p is not None:
            if not (0.0 <= top_p <= 1.0):
                raise ValueError("top_p must be between 0.0 and 1.0")
            self.top_p = top_p
            logger.info(f"Updated top_p to {top_p}")
        
        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            self.max_tokens = max_tokens
            logger.info(f"Updated max_tokens to {max_tokens}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("RESPONSE GENERATOR EXAMPLE")
    print("=" * 70)
    print("\nNote: Requires Ollama running with a model pulled.")
    print("Start Ollama with: ollama serve")
    print("Pull a model with: ollama pull mistral")
    print("\nSee the main app.py for complete usage example.\n")

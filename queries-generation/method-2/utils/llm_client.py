"""
LLM Client for Groq API
Handles all API calls with retry logic and rate limiting
"""

import os
import time
import logging
from typing import List, Dict, Optional
from groq import Groq

logger = logging.getLogger(__name__)

class GroqClient:
    """Wrapper for Groq API with retry logic and rate limiting"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.api_key)
        self.request_count = 0
        self.last_request_time = 0
        
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        max_retries: int = 3,
        retry_delay: int = 2
    ) -> Optional[str]:
        """Make a chat completion request to Groq API"""
        for attempt in range(max_retries):
            try:
                # Rate limiting
                elapsed = time.time() - self.last_request_time
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                
                self.last_request_time = time.time()
                self.request_count += 1
                
                content = response.choices[0].message.content
                return content.strip()
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}): {error_msg}")
                
                # Check if it's a rate limit error - give more info
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    logger.error(f"❌ RATE LIMIT HIT: {error_msg}")
                    logger.error("   Consider: reducing num_evolution_rounds or using cheaper model")
                elif "401" in error_msg or "authentication" in error_msg.lower():
                    logger.error(f"❌ AUTH ERROR: Check your GROQ_API_KEY in .env file")
                elif "model" in error_msg.lower():
                    logger.error(f"❌ MODEL ERROR: Model might not exist or be unavailable")
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"❌ All retries failed: {error_msg}")
                    return None
        
        return None
    
    def batch_completion(
        self,
        prompts: List[str],
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_message: str = None,
        show_progress: bool = True
    ) -> List[Optional[str]]:
        """Process multiple prompts with progress tracking"""
        results = []
        total = len(prompts)
        
        for i, prompt in enumerate(prompts):
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            result = self.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            results.append(result)
            
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{total} prompts")
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Get API usage statistics"""
        return {
            "total_requests": self.request_count,
            "elapsed_time": time.time() - self.last_request_time if self.request_count > 0 else 0
        }

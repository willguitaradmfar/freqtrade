#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI LLM client utility for freqtrade strategies
"""

import json
import logging
import os
import base64
from typing import Dict, List, Optional, Union, Any

# Configure logger
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    logger.info("OpenAI library successfully imported!")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not found. Please install with: pip install openai")

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64
    
    :param image_path: Path to the image file
    :return: Base64 encoded string or None if encoding fails
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

class LLMClient:
    """
    Generic OpenAI LLM client for sending messages and receiving responses
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM client
        
        :param api_key: OpenAI API key, if None will try to get from OPENAI_API_KEY env variable
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as a parameter or set OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("LLM client initialized")
    
    def send_message(self, 
                    messages: List[Dict], 
                    model: str = "gpt-4-turbo", 
                    temperature: float = 0.7,
                    max_tokens: int = 1024,
                    response_format: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Send messages to OpenAI and get response
        
        :param messages: List of message objects in OpenAI format
        :param model: OpenAI model to use
        :param temperature: Controls randomness (0-1)
        :param max_tokens: Maximum tokens in the response
        :param response_format: Optional format specification, e.g. {"type": "json_object"}
        
        :return: Dictionary with response details and parsed content
        """
        logger.info(f"🔄🔄🔄🔄🔄 Sending messages to OpenAI 🔄🔄🔄🔄🔄")

        if not messages:
            logger.error("No messages to send")
            return {
                "success": False,
                "error": "No messages to send",
                "content": None,
                "raw_response": None
            }
        
        try:
            # Prepare API call parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add response_format if specified
            if response_format:
                params["response_format"] = response_format
            
            # Call OpenAI API
            response = self.client.chat.completions.create(**params)
            
            # Parse response
            parsed_content = None
            raw_content = response.choices[0].message.content
            
            # Try to parse JSON if requested
            if response_format and response_format.get("type") == "json_object":
                try:
                    parsed_content = json.loads(raw_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    parsed_content = None
            
            return {
                "success": True,
                "model": model,
                "content": parsed_content or raw_content,
                "raw_response": raw_content,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "raw_response": None
            } 
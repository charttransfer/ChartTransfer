"""
LLM Client for calling Gemini Vision API.
"""

import os
import json
import re
import base64
import sys
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config


class LLMClient:
    """
    Unified LLM client.
    
    Calls the Gemini Vision API for multimodal analysis
    via an OpenAI-compatible interface.
    """
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
        """
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL
        )
        self.model_name = model_name
    
    def call_with_image(self, prompt: str, image_path: str, temperature: float = 0.2) -> Dict:
        """
        Call the LLM with a single image.
        
        Args:
            prompt: The prompt text
            image_path: Path to the image file
            temperature: Temperature parameter
            
        Returns:
            Parsed JSON result
        """
        return self.call_with_images(prompt, [image_path], temperature)
    
    def call_with_images(self, prompt: str, image_paths: List[str], temperature: float = 0.2) -> Dict:
        """
        Call the LLM with multiple images.
        
        Args:
            prompt: The prompt text
            image_paths: List of image file paths
            temperature: Temperature parameter
            
        Returns:
            Parsed JSON result
        """
        image_contents = []
        for img_path in image_paths:
            if img_path and os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })
        
        if not image_contents:
            raise ValueError("No valid images provided")
        
        # Build message content
        content = [{"type": "text", "text": prompt}] + image_contents
        
        # Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
            )
            response_text = response.choices[0].message.content.strip()
            if not response_text:
                print(f"⚠️  LLM returned empty response!")
                print(f"   Model: {self.model_name}")
                print(f"   Images: {len(image_contents)}")
        except Exception as e:
            print(f"❌ Error calling LLM: {e}")
            print(f"   Model: {self.model_name}")
            print(f"   Images: {image_paths}")
            raise
        
        # Parse the response
        return self._parse_json_response(response_text)
    
    def call_text_only(self, prompt: str, temperature: float = 0.2) -> Dict:
        """
        Call the LLM with text only (no images).
        
        Args:
            prompt: The prompt text
            temperature: Temperature parameter
            
        Returns:
            Parsed JSON result
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling LLM: {e}")
            raise
        
        return self._parse_json_response(response_text)
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON from the LLM response.
        
        Handles various possible formats:
        - Plain JSON
        - JSON wrapped in Markdown code blocks
        - JSON with extra surrounding text
        """
        text = response_text.strip()
        
        # Try to extract JSON code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to extract the first complete JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
        
        # Remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response text: {response_text[:500]}...")
            raise


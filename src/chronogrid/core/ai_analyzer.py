"""
Core AI Analyzer Module

Contains the core business logic for AI analysis operations.
Used by the APT pipeline stages.
"""

from pathlib import Path
from typing import Dict, Any, Optional

class AIAnalysisError(Exception):
    """Custom exception for AI analysis errors."""
    pass

class LlamaAIAnalyzer:
    """Core Llama AI analysis functionality."""

    def __init__(self, api_key: str, base_url: str = "https://api.llama.com/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = None
        self._init_session()

    def _init_session(self):
        """Initialize HTTP session for API calls."""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
        except ImportError:
            raise AIAnalysisError("requests library not available")

    def analyze_image(self, image_path: Path, prompt: str,
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Analyze image using Llama Vision API.

        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            max_tokens: Maximum response tokens

        Returns:
            Analysis result dictionary

        Raises:
            AIAnalysisError: On analysis failure
        """
        try:
            import base64

            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Prepare API request
            payload = {
                "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }

            # Make API request
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                raise AIAnalysisError(f"API request failed: {response.status_code} - {response.text}")

            result = response.json()

            # Extract analysis text
            analysis_text = ""
            if 'choices' in result and result['choices']:
                message = result['choices'][0].get('message', {})
                if 'content' in message:
                    analysis_text = message['content']

            return {
                'analysis_text': analysis_text,
                'api_response': result,
                'success': True,
                'tokens_used': result.get('usage', {}).get('total_tokens', 0)
            }

        except Exception as e:
            raise AIAnalysisError(f"Image analysis failed: {e}")

    def analyze_video_chronogrid(self, chronogrid_path: Path,
                                video_name: str) -> Dict[str, Any]:
        """
        Analyze chronogrid image with video-specific prompt.

        Args:
            chronogrid_path: Path to chronogrid image
            video_name: Name of original video

        Returns:
            Analysis result dictionary

        Raises:
            AIAnalysisError: On analysis failure
        """
        prompt = f"""Analyze this chronogrid visualization of video '{video_name}'.

A chronogrid shows video frames arranged in a grid, typically reading left-to-right, top-to-bottom chronologically.

Please describe:
1. What activities, events, or scenes are shown in the video
2. Any patterns, sequences, or notable changes over time
3. Key moments or transitions between different parts of the video
4. Overall theme or purpose of the video content
5. Any text, objects, or people visible in the frames

Be specific and detailed in your analysis."""

        return self.analyze_image(chronogrid_path, prompt)

class NetlifyProxyAnalyzer(LlamaAIAnalyzer):
    """AI analyzer using Netlify proxy endpoint."""

    def __init__(self):
        # Use Netlify proxy instead of direct API
        self.proxy_url = "https://llama-universal-netlify-project.netlify.app/.netlify/functions/llama-proxy?path=/chat/completions"
        self.session = None
        self._init_session()

    def _init_session(self):
        """Initialize session for proxy endpoint."""
        try:
            import requests
            import os

            self.session = requests.Session()
            api_key = os.environ.get('LLAMA_API_KEY')
            if not api_key:
                raise AIAnalysisError("LLAMA_API_KEY environment variable required")

            self.session.headers.update({
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            })
        except ImportError:
            raise AIAnalysisError("requests library not available")

    def analyze_image(self, image_path: Path, prompt: str,
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Analyze image using Netlify proxy.

        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            max_tokens: Maximum response tokens

        Returns:
            Analysis result dictionary

        Raises:
            AIAnalysisError: On analysis failure
        """
        try:
            import base64

            # Load and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Prepare API request for proxy
            payload = {
                "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }

            # Make API request to proxy
            response = self.session.post(
                self.proxy_url,
                json=payload,
                timeout=120
            )

            if response.status_code != 200:
                raise AIAnalysisError(f"Proxy request failed: {response.status_code} - {response.text}")

            result = response.json()

            # Extract analysis text
            analysis_text = ""
            if 'choices' in result and result['choices']:
                message = result['choices'][0].get('message', {})
                if 'content' in message:
                    analysis_text = message['content']

            return {
                'analysis_text': analysis_text,
                'api_response': result,
                'success': True,
                'proxy_used': True
            }

        except Exception as e:
            raise AIAnalysisError(f"Proxy analysis failed: {e}")

class AIAnalyzerFactory:
    """Factory for creating appropriate AI analyzer instances."""

    @staticmethod
    def create_analyzer(analyzer_type: str = "netlify") -> LlamaAIAnalyzer:
        """
        Create AI analyzer instance.

        Args:
            analyzer_type: Type of analyzer ("direct" or "netlify")

        Returns:
            Configured analyzer instance

        Raises:
            AIAnalysisError: On analyzer creation failure
        """
        if analyzer_type == "netlify":
            return NetlifyProxyAnalyzer()
        elif analyzer_type == "direct":
            import os
            api_key = os.environ.get('LLAMA_API_KEY')
            if not api_key:
                raise AIAnalysisError("LLAMA_API_KEY required for direct analyzer")
            return LlamaAIAnalyzer(api_key)
        else:
            raise AIAnalysisError(f"Unknown analyzer type: {analyzer_type}")

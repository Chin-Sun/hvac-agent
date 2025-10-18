"""
HVAC Booking Agent - LLM Client

OpenAI LLM client for processing HVAC booking conversations.

Author: Qian Sun
Date: 2025-10-17
Version: 1.0.0
License: MIT License

Copyright (c) 2025 Qian Sun. Licensed under the MIT License.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from .schema import BookingIntent, AgentOutput

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI LLM client"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize LLM client

        Args:
            api_key: OpenAI API key, if not provided, read from environment variable
            model: name of the model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly."
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def _chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call OpenAI chat completion API (internal method)

        Args:
            messages: list of messages
            temperature: temperature parameter, controls the randomness of the output
            max_tokens: maximum number of tokens

        Returns:
            content of the API response
        """
        try:
            logger.info(f"Calling OpenAI API with model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            logger.info("OpenAI API call successful")
            return content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise

    def process_conversation(
        self,
        conversation_turns: List[str],
        system_prompt: str,
        temperature: float = 0.1,
    ) -> AgentOutput:
        """
        Process complete conversation and return structured output

        Args:
            conversation_turns: list of conversation turns
            system_prompt: system prompt
            temperature: temperature parameter

        Returns:
            AgentOutput object, containing summary and booking information
        """
        # Build messages directly
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "\n".join(
                    [
                        f"Turn {i + 1}: {turn}"
                        for i, turn in enumerate(conversation_turns)
                    ]
                ),
            },
        ]

        try:
            # Call API
            response_content = self._chat_completion(messages, temperature=temperature)

            # Parse JSON response
            try:
                response_data = json.loads(response_content)

                # Validate and create AgentOutput
                agent_output = AgentOutput(
                    summary=response_data.get("summary", ""),
                    booking=BookingIntent(**response_data.get("booking", {})),
                )

                return agent_output

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                logger.error(f"Raw response: {response_content}")

                # Return error format AgentOutput
                return AgentOutput(
                    summary="Error: Failed to parse API response",
                    booking=BookingIntent(service_type="other", confidence=0.0),
                )

        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """
        Test API connection

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hello, this is a connection test."}
            ]

            self._chat_completion(test_messages, temperature=0.1, max_tokens=10)
            logger.info("API connection test successful")
            return True

        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False


# Convenience function
def create_llm_client(api_key: Optional[str] = None, model: str = "gpt-4") -> LLMClient:
    """
    Convenience function to create LLM client

    Args:
        api_key: OpenAI API key
        model: name of the model to use

    Returns:
        LLMClient instance
    """
    return LLMClient(api_key=api_key, model=model)

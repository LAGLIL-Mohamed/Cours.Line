"""Client API Mistral avec rate limiting et retry."""

import time
import os
from typing import Optional

from mistralai.client import Mistral


class MistralClient:
    """Client Mistral avec gestion du rate limiting et retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        min_delay: float = 1.5,
    ) -> None:
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Clé API Mistral requise. Définissez MISTRAL_API_KEY."
            )
        self.client = Mistral(api_key=self.api_key)
        self.model = model
        self.min_delay = min_delay
        self.last_call_time = 0.0

    def chat(
        self,
        messages: list,
        temperature: float = 0.3,
        max_retries: int = 3,
        force_json: bool = False,
    ) -> str:
        """Appel API avec rate limiting et retry exponentiel."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                response = self.client.chat.complete(**kwargs)
                self.last_call_time = time.time()
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                if (
                    "429" in err_str
                    or "rate limit" in err_str.lower()
                    or "too many requests" in err_str.lower()
                ):
                    wait = (2 ** attempt) + 2
                    print(f"  Rate limit atteint, attente {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Erreur API Mistral : {err_str}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        raise RuntimeError("Échec après max_retries")
"""LLM Client — OpenAI and Anthropic wrappers.

Config: llm_enabled: false by default.
Bot works fully without LLM — this is opt-in enhancement.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def is_llm_enabled() -> bool:
    """Check if LLM is enabled (config + API key present)."""
    import yaml
    config_path = "config/strategy_config.yaml"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        enabled = config.get("llm_enabled", False)
        if not enabled:
            return False
        # Check if any API key is available
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        return has_openai or has_anthropic
    except Exception:
        return False


class LLMClient:
    """Unified LLM client for OpenAI and Anthropic.

    Usage:
        client = LLMClient()
        response = client.complete("Analyze this market headline: ...")
    """

    def __init__(self, provider: str = "auto", model: Optional[str] = None):
        """Initialize LLM client.

        Args:
            provider: "openai", "anthropic", or "auto" (tries OpenAI first)
            model: Model to use (defaults to cheapest available)
        """
        self.provider = provider
        self.model = model
        self._client = None
        self._initialized = False

    def _init_client(self) -> bool:
        """Initialize the LLM client."""
        if self._initialized:
            return self._client is not None

        if self.provider in ("openai", "auto"):
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    import openai
                    self._client = openai.OpenAI(api_key=openai_key)
                    self.provider = "openai"
                    self.model = self.model or "gpt-4o-mini"
                    self._initialized = True
                    logger.info("LLM client: OpenAI (%s)", self.model)
                    return True
                except ImportError:
                    logger.warning("openai package not installed")

        if self.provider in ("anthropic", "auto"):
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                try:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=anthropic_key)
                    self.provider = "anthropic"
                    self.model = self.model or "claude-haiku-20240307"
                    self._initialized = True
                    logger.info("LLM client: Anthropic (%s)", self.model)
                    return True
                except ImportError:
                    logger.warning("anthropic package not installed")

        logger.warning("No LLM client available (check API keys and llm_enabled config)")
        self._initialized = True
        return False

    def complete(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Send prompt to LLM and return response.

        Returns None if LLM unavailable.
        """
        if not self._init_client() or self._client is None:
            return None

        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
                return response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

        except Exception as exc:
            logger.error("LLM completion failed: %s", exc)
            return None

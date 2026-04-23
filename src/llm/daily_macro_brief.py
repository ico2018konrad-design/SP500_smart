"""Daily Macro Brief — morning LLM analysis.

Runs at 07:00 CET. Analyzes 30-50 news headlines.
Returns: regime_bias, confidence, blackout_recommended.
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.llm.llm_client import LLMClient, is_llm_enabled

logger = logging.getLogger(__name__)


@dataclass
class MacroBrief:
    """Result of daily macro brief."""
    regime_bias: str           # "bullish", "neutral", "bearish"
    confidence: float          # 0.0-1.0
    blackout_recommended: bool
    key_themes: list
    summary: str
    timestamp: datetime
    llm_used: bool


MACRO_PROMPT_TEMPLATE = """You are a financial analyst. Analyze these market headlines and provide a brief assessment.

Headlines:
{headlines}

Respond ONLY in JSON format:
{{
  "regime_bias": "bullish/neutral/bearish",
  "confidence": 0.0-1.0,
  "blackout_recommended": true/false,
  "key_themes": ["theme1", "theme2"],
  "summary": "2-3 sentence summary"
}}

Criteria for blackout_recommended=true: FOMC meeting, CPI/NFP release today, geopolitical crisis.
"""


class DailyMacroBrief:
    """Morning macro brief using LLM analysis of headlines."""

    def __init__(self):
        self.client = LLMClient()

    def analyze(self, headlines: list) -> MacroBrief:
        """Analyze headlines and return macro brief.

        Args:
            headlines: List of news headline strings

        Returns:
            MacroBrief with regime assessment
        """
        if not is_llm_enabled() or not headlines:
            return self._fallback_brief()

        headline_text = "\n".join(f"- {h}" for h in headlines[:50])
        prompt = MACRO_PROMPT_TEMPLATE.format(headlines=headline_text)

        response = self.client.complete(prompt, max_tokens=300)

        if response is None:
            return self._fallback_brief()

        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return MacroBrief(
                    regime_bias=data.get("regime_bias", "neutral"),
                    confidence=float(data.get("confidence", 0.5)),
                    blackout_recommended=bool(data.get("blackout_recommended", False)),
                    key_themes=data.get("key_themes", []),
                    summary=data.get("summary", ""),
                    timestamp=datetime.now(),
                    llm_used=True,
                )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to parse LLM response: %s", exc)

        return self._fallback_brief()

    def _fallback_brief(self) -> MacroBrief:
        """Return neutral fallback when LLM unavailable."""
        return MacroBrief(
            regime_bias="neutral",
            confidence=0.5,
            blackout_recommended=False,
            key_themes=[],
            summary="LLM unavailable — using technical analysis only",
            timestamp=datetime.now(),
            llm_used=False,
        )

    def fetch_headlines(self) -> list:
        """Fetch current headlines from free sources.

        Returns list of headline strings.
        Note: This is a basic implementation using RSS feeds.
        """
        headlines = []
        try:
            import urllib.request
            from xml.etree import ElementTree

            rss_feeds = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            ]

            for feed_url in rss_feeds:
                try:
                    with urllib.request.urlopen(feed_url, timeout=5) as response:
                        content = response.read()
                    root = ElementTree.fromstring(content)
                    items = root.findall(".//item/title")
                    for item in items[:15]:
                        if item.text:
                            headlines.append(item.text.strip())
                except Exception:
                    continue

        except Exception as exc:
            logger.warning("Failed to fetch headlines: %s", exc)

        return headlines[:50]

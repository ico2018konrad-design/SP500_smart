"""Crash Narrative Detector — real-time panic detection.

Scans news every 30 minutes for systemic crisis language.
Triggers panic mode if confidence > 70%.
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.llm.llm_client import LLMClient, is_llm_enabled

logger = logging.getLogger(__name__)

PANIC_KEYWORDS = [
    "bank run", "bank failure", "fed emergency", "emergency cut",
    "war escalation", "nuclear", "systemic collapse", "margin calls",
    "circuit breaker", "trading halt", "flash crash", "contagion",
    "lehman", "bear stearns", "svb collapse", "credit crisis",
    "debt ceiling default", "sovereign default", "currency crisis",
]

CRASH_PROMPT_TEMPLATE = """You are a financial risk analyst. Analyze these headlines for signs of a systemic market crisis.

Headlines:
{headlines}

Known panic keywords: bank run, Fed emergency, war escalation, systemic collapse, bank failure.

Respond ONLY in JSON:
{{
  "panic_detected": true/false,
  "confidence": 0.0-1.0,
  "crisis_type": "banking/geopolitical/credit/none",
  "key_signals": ["signal1"],
  "recommendation": "normal/caution/panic"
}}
"""


@dataclass
class NarrativeResult:
    panic_detected: bool
    confidence: float
    crisis_type: str
    key_signals: list
    recommendation: str
    timestamp: datetime
    keyword_hits: list
    llm_used: bool


class CrashNarrativeDetector:
    """Detects systemic crisis narratives in news headlines."""

    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self.client = LLMClient()

    def analyze(self, headlines: list) -> NarrativeResult:
        """Analyze headlines for crash/panic narratives.

        Returns NarrativeResult with panic assessment.
        """
        # Fast keyword check first
        keyword_hits = []
        headlines_lower = [h.lower() for h in headlines]
        for keyword in PANIC_KEYWORDS:
            if any(keyword in h for h in headlines_lower):
                keyword_hits.append(keyword)

        # If keywords found, escalate to LLM
        if keyword_hits and is_llm_enabled():
            result = self._llm_analyze(headlines, keyword_hits)
            return result

        # Rule-based fallback
        panic_score = len(keyword_hits) / max(len(PANIC_KEYWORDS), 1)
        panic_detected = len(keyword_hits) >= 3

        return NarrativeResult(
            panic_detected=panic_detected,
            confidence=min(panic_score * 2, 1.0),
            crisis_type="unknown" if keyword_hits else "none",
            key_signals=keyword_hits,
            recommendation="panic" if panic_detected else "normal",
            timestamp=datetime.now(),
            keyword_hits=keyword_hits,
            llm_used=False,
        )

    def _llm_analyze(self, headlines: list, keyword_hits: list) -> NarrativeResult:
        """Use LLM for deeper analysis when keywords are detected."""
        headline_text = "\n".join(f"- {h}" for h in headlines[:30])
        prompt = CRASH_PROMPT_TEMPLATE.format(headlines=headline_text)

        response = self.client.complete(prompt, max_tokens=200)

        if response is None:
            return self._keyword_fallback(keyword_hits)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                confidence = float(data.get("confidence", 0.5))
                return NarrativeResult(
                    panic_detected=bool(data.get("panic_detected", False)) and
                                   confidence >= self.confidence_threshold,
                    confidence=confidence,
                    crisis_type=data.get("crisis_type", "none"),
                    key_signals=data.get("key_signals", keyword_hits),
                    recommendation=data.get("recommendation", "normal"),
                    timestamp=datetime.now(),
                    keyword_hits=keyword_hits,
                    llm_used=True,
                )
        except Exception as exc:
            logger.error("LLM crash analysis parse error: %s", exc)

        return self._keyword_fallback(keyword_hits)

    def _keyword_fallback(self, keyword_hits: list) -> NarrativeResult:
        confidence = min(len(keyword_hits) * 0.25, 1.0)
        return NarrativeResult(
            panic_detected=confidence >= self.confidence_threshold,
            confidence=confidence,
            crisis_type="unknown",
            key_signals=keyword_hits,
            recommendation="caution" if keyword_hits else "normal",
            timestamp=datetime.now(),
            keyword_hits=keyword_hits,
            llm_used=False,
        )

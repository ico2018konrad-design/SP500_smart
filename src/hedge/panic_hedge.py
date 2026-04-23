"""Panic Hedge Module — emergency mode.

Triggers (any one):
- VIX > 35
- SPY -5% in 2 sessions
- HY spreads +100bps in a week
- LLM detects systemic event (confidence > 70%)

Actions:
- Flat all LONG positions within 15 min
- Buy VXX (3% capital)
- Activate SPXS short (25% capital)
- Send Telegram alert
- Require manual unlock to exit panic mode
- Min 5 trading days in panic mode
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PanicTrigger:
    """Records what triggered panic mode."""
    trigger_type: str     # vix_spike, spy_crash, hy_spread, llm_narrative
    trigger_value: float
    threshold: float
    timestamp: datetime
    description: str


@dataclass
class PanicModeStatus:
    """Current panic mode state."""
    is_active: bool
    activated_at: Optional[datetime]
    triggers: List[PanicTrigger] = field(default_factory=list)
    manually_locked: bool = True  # requires manual unlock
    min_days_remaining: int = 5
    spxs_pct: float = 0.25
    vxx_pct: float = 0.03

    def can_deactivate(self, current_time: Optional[datetime] = None) -> bool:
        """Check if panic mode can be deactivated (min 5 trading days)."""
        if not self.is_active or self.activated_at is None:
            return False
        if self.manually_locked:
            return False
        if current_time is None:
            current_time = datetime.now()
        days_elapsed = (current_time - self.activated_at).days
        return days_elapsed >= self.min_days_remaining * 1.4  # ~7 calendar days for 5 trading days


class PanicHedge:
    """Emergency panic hedge module.

    Once activated, requires manual unlock + min 5 trading days.
    """

    def __init__(
        self,
        vix_threshold: float = 35.0,
        spy_2session_drop: float = -0.05,
        hy_weekly_bps: float = 100.0,
        llm_confidence: float = 0.70,
        vxx_pct: float = 0.03,
        spxs_pct: float = 0.25,
        min_panic_days: int = 5,
    ):
        self.vix_threshold = vix_threshold
        self.spy_2session_drop = spy_2session_drop
        self.hy_weekly_bps = hy_weekly_bps
        self.llm_confidence = llm_confidence
        self.vxx_pct = vxx_pct
        self.spxs_pct = spxs_pct
        self.min_panic_days = min_panic_days
        self.status = PanicModeStatus(is_active=False, activated_at=None)

    def check_triggers(
        self,
        vix: float,
        spy_2session_return: Optional[float] = None,
        hy_weekly_change_bps: Optional[float] = None,
        llm_panic_confidence: Optional[float] = None,
    ) -> Optional[PanicTrigger]:
        """Check if any panic trigger is activated.

        Returns PanicTrigger if any threshold breached, None otherwise.
        """
        # Trigger 1: VIX > 35
        if vix > self.vix_threshold:
            trigger = PanicTrigger(
                trigger_type="vix_spike",
                trigger_value=vix,
                threshold=self.vix_threshold,
                timestamp=datetime.now(),
                description=f"VIX {vix:.1f} > threshold {self.vix_threshold}",
            )
            logger.critical("PANIC TRIGGER: VIX %.1f > %.0f", vix, self.vix_threshold)
            return trigger

        # Trigger 2: SPY -5% in 2 sessions
        if spy_2session_return is not None and spy_2session_return <= self.spy_2session_drop:
            trigger = PanicTrigger(
                trigger_type="spy_crash",
                trigger_value=spy_2session_return,
                threshold=self.spy_2session_drop,
                timestamp=datetime.now(),
                description=f"SPY 2-session return {spy_2session_return:.1%}",
            )
            logger.critical(
                "PANIC TRIGGER: SPY 2-session %.1f%% <= %.0f%%",
                spy_2session_return * 100, self.spy_2session_drop * 100
            )
            return trigger

        # Trigger 3: HY spreads +100bps in a week
        if hy_weekly_change_bps is not None and hy_weekly_change_bps >= self.hy_weekly_bps:
            trigger = PanicTrigger(
                trigger_type="hy_spread",
                trigger_value=hy_weekly_change_bps,
                threshold=self.hy_weekly_bps,
                timestamp=datetime.now(),
                description=f"HY spreads +{hy_weekly_change_bps:.0f}bps in week",
            )
            logger.critical(
                "PANIC TRIGGER: HY spreads +%.0fbps >= %.0fbps",
                hy_weekly_change_bps, self.hy_weekly_bps
            )
            return trigger

        # Trigger 4: LLM systemic event detection
        if llm_panic_confidence is not None and llm_panic_confidence >= self.llm_confidence:
            trigger = PanicTrigger(
                trigger_type="llm_narrative",
                trigger_value=llm_panic_confidence,
                threshold=self.llm_confidence,
                timestamp=datetime.now(),
                description=f"LLM systemic event confidence {llm_panic_confidence:.1%}",
            )
            logger.critical(
                "PANIC TRIGGER: LLM narrative confidence %.1f%%",
                llm_panic_confidence * 100
            )
            return trigger

        return None

    def activate(self, trigger: PanicTrigger) -> PanicModeStatus:
        """Activate panic mode.

        This will:
        1. Set panic mode active
        2. Record trigger
        3. Require manual unlock to deactivate
        """
        if self.status.is_active:
            self.status.triggers.append(trigger)
            return self.status

        self.status = PanicModeStatus(
            is_active=True,
            activated_at=datetime.now(),
            triggers=[trigger],
            manually_locked=True,
            min_days_remaining=self.min_panic_days,
            spxs_pct=self.spxs_pct,
            vxx_pct=self.vxx_pct,
        )

        logger.critical(
            "PANIC MODE ACTIVATED: %s | Min %d trading days | "
            "SPXS %.0f%% | VXX %.0f%% | REQUIRES MANUAL UNLOCK",
            trigger.description,
            self.min_panic_days,
            self.spxs_pct * 100,
            self.vxx_pct * 100,
        )

        return self.status

    def manual_unlock(self) -> bool:
        """Manually unlock panic mode (allows deactivation after min days)."""
        if not self.status.is_active:
            return False

        self.status.manually_locked = False
        logger.warning("Panic mode manually UNLOCKED — can deactivate after min days")
        return True

    def deactivate(self) -> bool:
        """Deactivate panic mode if conditions are met."""
        if not self.status.can_deactivate():
            logger.warning(
                "Cannot deactivate panic mode: locked=%s, days_active=%s",
                self.status.manually_locked,
                (datetime.now() - self.status.activated_at).days
                if self.status.activated_at else "N/A",
            )
            return False

        self.status.is_active = False
        logger.info("Panic mode DEACTIVATED")
        return True

    def get_allocations(self, capital: float) -> dict:
        """Get required allocations during panic mode."""
        if not self.status.is_active:
            return {}

        return {
            "SPXS": capital * self.spxs_pct,
            "VXX": capital * self.vxx_pct,
            "cash_remaining": capital * (1 - self.spxs_pct - self.vxx_pct),
        }

    def save_state(self, filepath: str) -> None:
        """Save panic mode state to file for persistence across restarts."""
        import os
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        state = {
            "is_active": self.status.is_active,
            "activated_at": self.status.activated_at.isoformat() if self.status.activated_at else None,
            "manually_locked": self.status.manually_locked,
            "triggers": [
                {
                    "trigger_type": t.trigger_type,
                    "trigger_value": t.trigger_value,
                    "threshold": t.threshold,
                    "timestamp": t.timestamp.isoformat(),
                    "description": t.description,
                }
                for t in self.status.triggers
            ],
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> bool:
        """Load panic mode state from file."""
        import os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath) as f:
                state = json.load(f)
            self.status.is_active = state.get("is_active", False)
            if state.get("activated_at"):
                self.status.activated_at = datetime.fromisoformat(state["activated_at"])
            self.status.manually_locked = state.get("manually_locked", True)
            return True
        except Exception as exc:
            logger.error("Failed to load panic state: %s", exc)
            return False

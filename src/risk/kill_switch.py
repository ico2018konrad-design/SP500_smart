"""Kill Switch — manual and automatic halt mechanisms."""
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

KILL_SWITCH_FILE = "kill_switch.json"


@dataclass
class KillSwitchState:
    is_active: bool = False
    activated_at: Optional[str] = None
    reason: str = ""
    activated_by: str = "auto"   # "manual" or "auto"
    requires_manual_reset: bool = False


class KillSwitch:
    """Global kill switch for the trading bot.

    Can be activated:
    - Manually (operator intervention)
    - Automatically (circuit breakers, API errors, systemic events)

    Manual activation always requires manual reset.
    """

    def __init__(self, state_file: str = KILL_SWITCH_FILE):
        self.state_file = state_file
        self.state = KillSwitchState()
        self._load_state()

    def is_active(self) -> bool:
        """Check if kill switch is active."""
        self._load_state()  # Always read from file for real-time updates
        return self.state.is_active

    def activate(
        self,
        reason: str = "Manual activation",
        activated_by: str = "manual",
        requires_manual_reset: bool = True,
    ) -> None:
        """Activate kill switch."""
        self.state = KillSwitchState(
            is_active=True,
            activated_at=datetime.now().isoformat(),
            reason=reason,
            activated_by=activated_by,
            requires_manual_reset=requires_manual_reset,
        )
        self._save_state()
        logger.critical(
            "KILL SWITCH ACTIVATED: %s (by=%s, manual_reset=%s)",
            reason, activated_by, requires_manual_reset
        )

    def deactivate(self, force: bool = False) -> bool:
        """Deactivate kill switch.

        Args:
            force: If True, override manual reset requirement

        Returns:
            True if deactivated, False if blocked
        """
        if self.state.requires_manual_reset and not force:
            logger.warning(
                "Kill switch requires manual reset. Use force=True or touch %s",
                self.state_file
            )
            return False

        self.state = KillSwitchState(is_active=False)
        self._save_state()
        logger.info("Kill switch deactivated")
        return True

    def _save_state(self) -> None:
        """Save kill switch state to file."""
        try:
            data = {
                "is_active": self.state.is_active,
                "activated_at": self.state.activated_at,
                "reason": self.state.reason,
                "activated_by": self.state.activated_by,
                "requires_manual_reset": self.state.requires_manual_reset,
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save kill switch state: %s", exc)

    def _load_state(self) -> None:
        """Load kill switch state from file."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            self.state = KillSwitchState(
                is_active=data.get("is_active", False),
                activated_at=data.get("activated_at"),
                reason=data.get("reason", ""),
                activated_by=data.get("activated_by", "auto"),
                requires_manual_reset=data.get("requires_manual_reset", False),
            )
        except Exception as exc:
            logger.error("Failed to load kill switch state: %s", exc)

    def get_status(self) -> dict:
        """Get kill switch status."""
        self._load_state()
        return {
            "is_active": self.state.is_active,
            "activated_at": self.state.activated_at,
            "reason": self.state.reason,
            "activated_by": self.state.activated_by,
            "requires_manual_reset": self.state.requires_manual_reset,
        }

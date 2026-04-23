"""Event Filter — earnings and macro calendar awareness.

Scans weekly for major events and applies pre-blackout 24h before.
Major events: FOMC, CPI, NFP, earnings of major S&P 500 components.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketEvent:
    event_type: str
    name: str
    datetime_utc: datetime
    importance: str   # HIGH, MEDIUM, LOW
    blackout_hours: int = 24


class EventFilter:
    """Manages pre-event blackout periods."""

    # Known recurring FOMC meeting dates (approximate)
    # In production, use an economic calendar API
    KNOWN_FOMC_MONTHS = [1, 3, 5, 6, 7, 9, 10, 11]  # approximate monthly schedule

    def __init__(self, blackout_hours: int = 24):
        self.blackout_hours = blackout_hours
        self._upcoming_events: List[MarketEvent] = []

    def add_event(self, event: MarketEvent) -> None:
        """Add an event to the calendar."""
        self._upcoming_events.append(event)
        # Keep only future events
        now = datetime.now()
        self._upcoming_events = [
            e for e in self._upcoming_events
            if e.datetime_utc > now - timedelta(hours=24)
        ]

    def is_in_blackout(self, check_time: Optional[datetime] = None) -> tuple:
        """Check if current time is within a blackout period.

        Returns (in_blackout, event_name)
        """
        if check_time is None:
            check_time = datetime.now()

        for event in self._upcoming_events:
            blackout_start = event.datetime_utc - timedelta(hours=event.blackout_hours)
            blackout_end = event.datetime_utc + timedelta(hours=2)

            if blackout_start <= check_time <= blackout_end:
                return True, event.name

        return False, ""

    def get_upcoming_events(self, hours_ahead: int = 48) -> List[MarketEvent]:
        """Get events in the next N hours."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)
        return [
            e for e in self._upcoming_events
            if now <= e.datetime_utc <= cutoff
        ]

    def load_fomc_calendar(self, year: int = None) -> None:
        """Load approximate FOMC calendar.

        In production, use economiccalendar.com API or scraped data.
        This is a simplified version.
        """
        if year is None:
            year = datetime.now().year

        # Approximate FOMC dates (second Wednesday of scheduled months)
        for month in self.KNOWN_FOMC_MONTHS:
            try:
                # Find second Wednesday
                first_day = datetime(year, month, 1)
                wednesday = 2  # Wednesday = weekday index 2
                days_to_wed = (wednesday - first_day.weekday()) % 7
                first_wed = first_day + timedelta(days=days_to_wed)
                second_wed = first_wed + timedelta(weeks=1)

                self.add_event(MarketEvent(
                    event_type="FOMC",
                    name=f"FOMC Meeting {year}-{month:02d}",
                    datetime_utc=second_wed.replace(hour=18, minute=0),  # ~2pm ET = 18 UTC
                    importance="HIGH",
                    blackout_hours=24,
                ))
            except ValueError:
                continue

    def is_major_event_today(self) -> bool:
        """Quick check if any major event is today."""
        in_blackout, _ = self.is_in_blackout()
        return in_blackout

    def get_next_blackout(self) -> Optional[MarketEvent]:
        """Get the next upcoming blackout event."""
        now = datetime.now()
        upcoming = sorted(
            [e for e in self._upcoming_events if e.datetime_utc > now],
            key=lambda e: e.datetime_utc,
        )
        return upcoming[0] if upcoming else None

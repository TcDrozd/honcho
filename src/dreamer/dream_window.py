"""Time-of-day window that gates when automatic dreams may start."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import settings


def seconds_until_dream_window(now: datetime) -> float:
    """Return how long until automatic dreams may start.

    Args:
        now: The current time. Must be timezone-aware.

    Returns:
        0 when no window is configured or `now` is inside it; otherwise the
        seconds until the window next opens.
    """
    start = settings.DREAM.WINDOW_START_HOUR
    end = settings.DREAM.WINDOW_END_HOUR
    if start is None or end is None:
        return 0.0

    local = now.astimezone(ZoneInfo(settings.DREAM.WINDOW_TIMEZONE))
    if start < end:
        inside = start <= local.hour < end
    else:  # wraps midnight
        inside = local.hour >= start or local.hour < end
    if inside:
        return 0.0

    opens = local.replace(hour=start, minute=0, second=0, microsecond=0)
    if opens <= local:
        opens += timedelta(days=1)
    # Compare in UTC: subtracting two datetimes that share a ZoneInfo ignores
    # their offsets, which is off by an hour across a DST change.
    return (opens.astimezone(UTC) - now.astimezone(UTC)).total_seconds()

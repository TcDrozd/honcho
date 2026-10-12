"""Tests for the time-of-day window that gates automatic dreams."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import DreamSettings
from src.dreamer.dream_scheduler import DreamScheduler, check_and_schedule_dream
from src.dreamer.dream_window import seconds_until_dream_window
from src.schemas import DreamType


@contextmanager
def _window(
    start: int | None, end: int | None, timezone: str = "UTC"
) -> Generator[None]:
    with (
        patch("src.dreamer.dream_window.settings.DREAM.WINDOW_START_HOUR", start),
        patch("src.dreamer.dream_window.settings.DREAM.WINDOW_END_HOUR", end),
        patch("src.dreamer.dream_window.settings.DREAM.WINDOW_TIMEZONE", timezone),
    ):
        yield


def _utc(hour: int, minute: int = 0, day: int = 12, month: int = 10) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=UTC)


class TestSecondsUntilDreamWindow:
    def test_no_window_is_always_open(self):
        with _window(None, None):
            assert seconds_until_dream_window(_utc(14)) == 0

    def test_inside_window(self):
        with _window(1, 6):
            assert seconds_until_dream_window(_utc(1)) == 0
            assert seconds_until_dream_window(_utc(5, 59)) == 0

    def test_end_hour_is_exclusive(self):
        with _window(1, 6):
            assert seconds_until_dream_window(_utc(6)) == 19 * 3600

    def test_before_window_opens_same_day(self):
        with _window(1, 6):
            assert seconds_until_dream_window(_utc(0, 30)) == 30 * 60

    def test_after_window_waits_until_next_day(self):
        with _window(1, 6):
            assert seconds_until_dream_window(_utc(14)) == 11 * 3600

    def test_window_wrapping_midnight(self):
        with _window(22, 6):
            assert seconds_until_dream_window(_utc(23)) == 0
            assert seconds_until_dream_window(_utc(3)) == 0
            assert seconds_until_dream_window(_utc(21)) == 3600
            assert seconds_until_dream_window(_utc(6)) == 16 * 3600

    def test_hours_are_in_the_configured_timezone(self):
        # 14:00 UTC is 10:00 in New York (EDT, UTC-4); 01:00 local is 05:00 UTC.
        with _window(1, 6, "America/New_York"):
            assert seconds_until_dream_window(_utc(14)) == 15 * 3600
            assert seconds_until_dream_window(_utc(6)) == 0

    def test_wait_spans_dst_change(self):
        # Fall back (Nov 1): 12:00 EDT Oct 31 (16:00 UTC) → 01:00 EDT Nov 1
        # (05:00 UTC) = 13h, before the 02:00 EDT → 01:00 EST change.
        with _window(1, 6, "America/New_York"):
            assert seconds_until_dream_window(_utc(16, day=31)) == 13 * 3600
        # Spring forward: 12:00 EST Mar 7 (17:00 UTC) → 03:00 EDT Mar 8
        # (07:00 UTC) = 14h, with a window opening at 03:00.
        with _window(3, 6, "America/New_York"):
            assert seconds_until_dream_window(_utc(17, day=7, month=3)) == 14 * 3600


class TestDreamWindowSettings:
    def test_defaults_to_no_window(self):
        settings = DreamSettings()
        assert settings.WINDOW_START_HOUR is None
        assert settings.WINDOW_END_HOUR is None

    def test_start_and_end_must_be_set_together(self):
        with pytest.raises(ValidationError, match="must be set together"):
            DreamSettings(WINDOW_START_HOUR=1)

    def test_start_and_end_must_differ(self):
        with pytest.raises(ValidationError, match="must differ"):
            DreamSettings(WINDOW_START_HOUR=3, WINDOW_END_HOUR=3)

    def test_unknown_timezone_is_rejected(self):
        with pytest.raises(ValidationError, match="IANA timezone"):
            DreamSettings(
                WINDOW_START_HOUR=1,
                WINDOW_END_HOUR=6,
                WINDOW_TIMEZONE="Mars/Olympus_Mons",
            )

    def test_hour_out_of_range_is_rejected(self):
        with pytest.raises(ValidationError):
            DreamSettings(WINDOW_START_HOUR=24, WINDOW_END_HOUR=6)


@pytest.fixture
def dream_scheduler():
    DreamScheduler.reset_singleton()
    scheduler = DreamScheduler()
    with patch(
        "src.dreamer.dream_scheduler.get_dream_scheduler", return_value=scheduler
    ):
        yield scheduler
    DreamScheduler.reset_singleton()


@pytest.mark.asyncio
class TestDeriverSchedulerWindow:
    @pytest.fixture(autouse=True)
    def _pin_dream_config(self):
        with (
            patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED", True),
            patch("src.dreamer.dream_scheduler.settings.DREAM.DOCUMENT_THRESHOLD", 50),
            patch("src.dreamer.dream_scheduler.settings.DREAM.ENABLED_TYPES", ["omni"]),
            patch(
                "src.dreamer.dream_scheduler.settings.DREAM.IDLE_TIMEOUT_MINUTES", 60
            ),
            patch("src.dreamer.dream_scheduler.settings.DERIVER.SCHEDULER", "deriver"),
        ):
            yield

    async def _due_collection(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ) -> models.Collection:
        workspace, peer = sample_data
        collection = models.Collection(
            observer=peer.name,
            observed=peer.name,
            workspace_name=workspace.name,
            internal_metadata={},
        )
        db_session.add(collection)
        for _ in range(60):
            db_session.add(
                models.Document(
                    content="test",
                    level="explicit",
                    workspace_name=workspace.name,
                    observer=peer.name,
                    observed=peer.name,
                )
            )
        await db_session.commit()
        return collection

    async def test_outside_window_delays_until_it_opens(
        self,
        dream_scheduler: DreamScheduler,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await self._due_collection(db_session, sample_data)

        with (
            patch(
                "src.dreamer.dream_scheduler.seconds_until_dream_window",
                return_value=5 * 3600 + 1,
            ),
            patch.object(
                dream_scheduler, "schedule_dream", new_callable=AsyncMock
            ) as mock_schedule,
        ):
            assert await check_and_schedule_dream(db_session, collection) is True

        delay_minutes = mock_schedule.call_args.args[2]
        assert delay_minutes == 5 * 60 + 1
        assert mock_schedule.call_args.kwargs["delay_reason"] == "dream_window"

    async def test_window_shorter_than_idle_timeout_keeps_idle_timeout(
        self,
        dream_scheduler: DreamScheduler,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await self._due_collection(db_session, sample_data)

        with (
            patch(
                "src.dreamer.dream_scheduler.seconds_until_dream_window",
                return_value=10 * 60,
            ),
            patch.object(
                dream_scheduler, "schedule_dream", new_callable=AsyncMock
            ) as mock_schedule,
        ):
            assert await check_and_schedule_dream(db_session, collection) is True

        assert mock_schedule.call_args.args[2] == 60
        assert mock_schedule.call_args.kwargs["delay_reason"] == "idle_timeout"

    async def test_idle_timer_ending_outside_window_waits_for_it(
        self, dream_scheduler: DreamScheduler
    ):
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with (
            patch("src.dreamer.dream_scheduler.asyncio.sleep", fake_sleep),
            patch(
                "src.dreamer.dream_scheduler.seconds_until_dream_window",
                return_value=7200.0,
            ),
            patch.object(
                dream_scheduler, "execute_dream", new_callable=AsyncMock
            ) as mock_execute,
        ):
            await dream_scheduler._delayed_dream(  # pyright: ignore[reportPrivateUsage]
                "wuk",
                "ws",
                60,
                DreamType.OMNI,
                observer="p",
                observed="p",
                delay_reason="idle_timeout",
            )

        assert sleeps == [3600, 7200.0]
        assert mock_execute.call_args.kwargs["delay_reason"] == "dream_window"

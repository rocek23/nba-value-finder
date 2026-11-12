from datetime import date, datetime, timezone

from nba_vf.ui.live import nba_dates


def test_nba_dates_use_new_york_day_not_utc():
    # 12/11 00:25 UTC = 01:25 in Italia = 19:25 a New York dell'11/11: le partite in corso sono dell'11/11
    now = datetime(2026, 11, 12, 0, 25, tzinfo=timezone.utc)
    assert nba_dates(now, 1) == [date(2026, 11, 11), date(2026, 11, 12)]


def test_nba_dates_daytime_and_days_ahead():
    now = datetime(2026, 11, 12, 15, 0, tzinfo=timezone.utc)  # 10:00 a New York
    assert nba_dates(now, 0) == [date(2026, 11, 12)]
    assert nba_dates(now, 2) == [date(2026, 11, 12), date(2026, 11, 13), date(2026, 11, 14)]

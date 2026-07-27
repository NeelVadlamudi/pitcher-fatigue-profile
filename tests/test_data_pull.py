from datetime import date

import pandas as pd
import pytest

from pitcher_fatigue.data_pull import (
    find_cached_season,
    load_or_pull,
    monthly_date_chunks,
    season_date_bounds,
)


def test_monthly_chunks_are_inclusive_and_non_overlapping():
    chunks = list(monthly_date_chunks(date(2024, 3, 29), date(2024, 5, 2)))
    assert chunks == [
        (date(2024, 3, 29), date(2024, 3, 31)),
        (date(2024, 4, 1), date(2024, 4, 30)),
        (date(2024, 5, 1), date(2024, 5, 2)),
    ]


def test_current_season_does_not_query_future_dates():
    start, end = season_date_bounds(2026, today=date(2026, 7, 27))
    assert start == date(2026, 3, 1)
    assert end == date(2026, 7, 27)


def test_statcast_precoverage_season_is_rejected():
    with pytest.raises(ValueError):
        season_date_bounds(2014, today=date(2026, 7, 27))


def test_cached_season_is_loaded_without_network_lookup(tmp_path, monkeypatch):
    path = tmp_path / "webb_logan_657277_2024.csv"
    pd.DataFrame({"game_pk": [1], "release_speed": [92.5]}).to_csv(
        path,
        index=False,
    )

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("network lookup should not run for an exact cache hit")

    monkeypatch.setattr(
        "pitcher_fatigue.data_pull.get_pitcher_id",
        unexpected_lookup,
    )
    frame, provenance = load_or_pull("Logan", "Webb", 2024, cache_dir=tmp_path)

    assert frame["release_speed"].tolist() == [92.5]
    assert provenance["pitcher_id"] == 657277
    assert provenance["source"] == "local_cache"
    assert provenance["upstream_source"] == "baseball_savant_via_pybaseball"
    assert provenance["cached_at"].endswith("+00:00")
    assert find_cached_season("Logan", "Webb", 2024, tmp_path) == (
        path,
        657277,
    )

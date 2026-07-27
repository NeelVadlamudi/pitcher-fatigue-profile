"""Retrieve and cache pitcher-specific Statcast data.

This is the only production module that contacts the internet. All downstream
analysis accepts local ``pandas.DataFrame`` objects.
"""

from __future__ import annotations

import calendar
import re
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


class DataPullError(RuntimeError):
    """Raised when a Statcast request cannot produce usable data."""


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_value.lower()).strip("_")


def _pybaseball_functions(
    cache_dir: str | Path | None = None,
) -> tuple[Callable, Callable]:
    try:
        from pybaseball import cache, playerid_lookup, statcast_pitcher
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise DataPullError(
            "pybaseball is unavailable. Install the project requirements before "
            "requesting live Statcast data."
        ) from exc

    if cache_dir is not None:
        package_cache = Path(cache_dir) / ".pybaseball"
        package_cache.mkdir(parents=True, exist_ok=True)
        cache.config.cache_directory = str(package_cache)
    cache.enable()
    return playerid_lookup, statcast_pitcher


def get_pitcher_id(
    first_name: str,
    last_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> int:
    """Return the most recent exact-name MLBAM pitcher ID."""

    if not first_name.strip() or not last_name.strip():
        raise ValueError("Both first_name and last_name are required")

    playerid_lookup, _ = _pybaseball_functions(cache_dir)
    results = playerid_lookup(last_name.strip(), first_name.strip())
    if results.empty:
        raise DataPullError(f"No MLB player found for {first_name} {last_name}")

    normalized_first = first_name.strip().casefold()
    normalized_last = last_name.strip().casefold()
    exact = results[
        results["name_first"].astype(str).str.casefold().eq(normalized_first)
        & results["name_last"].astype(str).str.casefold().eq(normalized_last)
    ]
    candidates = exact if not exact.empty else results

    if "mlb_played_last" in candidates:
        candidates = candidates.sort_values(
            ["mlb_played_last", "key_mlbam"],
            ascending=[False, False],
            na_position="last",
        )
    return int(candidates.iloc[0]["key_mlbam"])


def season_date_bounds(season: int, today: date | None = None) -> tuple[date, date]:
    """Return a broad MLB date window that is later filtered to regular season."""

    today = today or date.today()
    if season < 2015:
        raise ValueError("Statcast pitch-level coverage begins in 2015")
    if season > today.year:
        raise ValueError(f"Season {season} is in the future")

    start = date(season, 3, 1)
    end = date(season, 11, 30)
    if season == today.year:
        end = min(end, today)
    return start, end


def monthly_date_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Yield inclusive, non-overlapping monthly request windows."""

    cursor = start
    while cursor <= end:
        month_end = date(
            cursor.year,
            cursor.month,
            calendar.monthrange(cursor.year, cursor.month)[1],
        )
        chunk_end = min(month_end, end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def cache_path(
    first_name: str,
    last_name: str,
    season: int,
    pitcher_id: int,
    cache_dir: str | Path = "data/raw",
) -> Path:
    filename = (
        f"{_slugify(last_name)}_{_slugify(first_name)}_{pitcher_id}_{season}.csv"
    )
    return Path(cache_dir) / filename


def find_cached_season(
    first_name: str,
    last_name: str,
    season: int,
    cache_dir: str | Path = "data/raw",
) -> tuple[Path, int] | None:
    """Find the newest exact-name cache without making a player-lookup request."""

    directory = Path(cache_dir)
    prefix = f"{_slugify(last_name)}_{_slugify(first_name)}_"
    pattern = f"{prefix}*_{season}.csv"
    matches = sorted(
        directory.glob(pattern),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for candidate in matches:
        middle = candidate.name.removeprefix(prefix).removesuffix(f"_{season}.csv")
        if middle.isdigit():
            return candidate, int(middle)
    return None


def pull_season(
    pitcher_id: int,
    season: int,
    *,
    pause_seconds: float = 0.5,
    request_fn: Callable | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Pull one pitcher's season in bounded monthly chunks.

    A broad March-November window prevents late-March regular-season games from
    being omitted. Postseason and spring-training rows are removed downstream
    using ``game_type``.
    """

    if request_fn is None:
        _, request_fn = _pybaseball_functions(cache_dir)

    start, end = season_date_bounds(season)
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for chunk_start, chunk_end in monthly_date_chunks(start, end):
        try:
            frame = request_fn(
                chunk_start.isoformat(),
                chunk_end.isoformat(),
                player_id=int(pitcher_id),
            )
        except Exception as exc:  # pragma: no cover - live-source behavior
            errors.append(f"{chunk_start:%Y-%m}: {exc}")
            continue
        if frame is not None and not frame.empty:
            frames.append(frame)
        if pause_seconds:
            time.sleep(pause_seconds)

    if not frames:
        detail = f" Errors: {'; '.join(errors)}" if errors else ""
        raise DataPullError(
            f"No Statcast data returned for pitcher {pitcher_id} in {season}.{detail}"
        )

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates().reset_index(drop=True)
    return result


def load_or_pull(
    first_name: str,
    last_name: str,
    season: int,
    *,
    cache_dir: str | Path = "data/raw",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a cached CSV or retrieve it from Statcast.

    Returns the data and provenance metadata used by the app and notebooks.
    """

    cached = (
        None
        if force_refresh
        else find_cached_season(first_name, last_name, season, cache_dir)
    )
    if cached is not None:
        path, pitcher_id = cached
        frame = pd.read_csv(path, low_memory=False)
        source = "local_cache"
    else:
        pitcher_id = get_pitcher_id(
            first_name,
            last_name,
            cache_dir=cache_dir,
        )
        path = cache_path(first_name, last_name, season, pitcher_id, cache_dir)
        frame = pull_season(pitcher_id, season, cache_dir=cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        source = "baseball_savant_via_pybaseball"

    provenance = {
        "pitcher_name": f"{first_name.strip()} {last_name.strip()}",
        "pitcher_id": pitcher_id,
        "season": season,
        "source": source,
        "upstream_source": "baseball_savant_via_pybaseball",
        "cache_path": str(path),
        "cached_at": datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )
        .replace(microsecond=0)
        .isoformat(),
        "rows": int(len(frame)),
    }
    return frame, provenance

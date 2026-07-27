"""Data-quality checks for Statcast pitch-level records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .config import AnalysisConfig


Severity = Literal["critical", "high", "medium", "low"]

REQUIRED_COLUMNS = {
    "game_date",
    "game_pk",
    "inning",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "release_speed",
}

NUMERIC_COLUMNS = {
    "game_pk",
    "inning",
    "at_bat_number",
    "pitch_number",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
}

PITCH_KEY = ["game_pk", "at_bat_number", "pitch_number"]

TRACKED_FIELDS = {
    "velocity": "release_speed",
    "spin_rate": "release_spin_rate",
    "extension": "release_extension",
    "horizontal_break": "pfx_x",
    "vertical_break": "pfx_z",
}


def field_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """Return observation coverage for each tracked pitch measurement."""

    return {
        label: (
            float(frame[column].notna().mean())
            if column in frame and len(frame)
            else 0.0
        )
        for label, column in TRACKED_FIELDS.items()
    }


def mnar_risk(
    frame: pd.DataFrame,
    field: str = "release_spin_rate",
    count_col: str = "game_pitch_count",
    early_cutoff: int = 30,
    late_cutoff: int = 80,
    min_pitches: int = 50,
) -> float | None:
    """Return late-minus-early missingness after minimum-window guards.

    The diagnostic compares pitch rows directly so sparse tail pitch counts do
    not receive the same weight as well-observed counts. A positive value means
    the field is missing more often late in games. It does not establish why.
    """

    if field not in frame or count_col not in frame:
        return None
    counts = pd.to_numeric(frame[count_col], errors="coerce")
    early = frame.loc[counts.le(early_cutoff), field]
    late = frame.loc[counts.ge(late_cutoff), field]
    if len(early) < min_pitches or len(late) < min_pitches:
        return None
    return float(late.isna().mean() - early.isna().mean())


@dataclass(frozen=True)
class QualityIssue:
    check: str
    severity: Severity
    evidence: str
    impact: str
    remediation: str


@dataclass
class DataQualityReport:
    rows_input: int
    rows_clean: int = 0
    games: int = 0
    eligible_starts: int = 0
    source_as_of: str | None = None
    issues: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)

    @property
    def status(self) -> str:
        severities = {issue.severity for issue in self.issues}
        if "critical" in severities:
            return "blocked"
        if severities & {"high", "medium"}:
            return "usable_with_caveats"
        return "ready"

    @property
    def is_usable(self) -> bool:
        return self.status != "blocked"

    def to_frame(self) -> pd.DataFrame:
        columns = ["check", "severity", "evidence", "impact", "remediation"]
        if not self.issues:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([asdict(issue) for issue in self.issues], columns=columns)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "rows_input": self.rows_input,
            "rows_clean": self.rows_clean,
            "games": self.games,
            "eligible_starts": self.eligible_starts,
            "source_as_of": self.source_as_of,
            "metrics": self.metrics,
            "issues": [asdict(issue) for issue in self.issues],
        }


class DataQualityError(ValueError):
    """Raised when required Statcast fields are missing."""

    def __init__(self, message: str, report: DataQualityReport):
        super().__init__(message)
        self.report = report


def _add_issue(
    report: DataQualityReport,
    check: str,
    severity: Severity,
    evidence: str,
    impact: str,
    remediation: str,
) -> None:
    report.issues.append(
        QualityIssue(
            check=check,
            severity=severity,
            evidence=evidence,
            impact=impact,
            remediation=remediation,
        )
    )


def validate_and_clean_statcast(
    frame: pd.DataFrame,
    config: AnalysisConfig | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Validate schema, grain, ranges, and starter coverage.

    Invalid physical measurements are set to missing rather than clipped.
    Duplicate pitch keys are retained once and surfaced as a quality issue.
    """

    config = config or AnalysisConfig()
    config.validate()
    report = DataQualityReport(rows_input=int(len(frame)))
    data = frame.copy()

    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        _add_issue(
            report,
            "required_columns",
            "critical",
            f"Missing columns: {', '.join(missing)}",
            "Pitch ordering or velocity degradation cannot be computed.",
            "Re-pull the data from Statcast with pitch-level detail fields.",
        )
        raise DataQualityError("Required Statcast columns are missing", report)

    exact_duplicates = int(data.duplicated().sum())
    if exact_duplicates:
        data = data.drop_duplicates().copy()
        _add_issue(
            report,
            "exact_duplicates",
            "medium",
            f"{exact_duplicates:,} exact duplicate rows removed.",
            "Unremoved duplicates would overweight affected pitches.",
            "Retain the ingestion deduplication check.",
        )

    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    for column in sorted(NUMERIC_COLUMNS & set(data.columns)):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    missing_key = data[PITCH_KEY].isna().any(axis=1)
    missing_key_count = int(missing_key.sum())
    if missing_key_count:
        data = data.loc[~missing_key].copy()
        _add_issue(
            report,
            "pitch_key_completeness",
            "high",
            f"{missing_key_count:,} rows with incomplete pitch keys removed.",
            "Those rows cannot be placed reliably within a game.",
            "Inspect the raw request and Statcast schema for malformed records.",
        )

    invalid_dates = data["game_date"].isna()
    invalid_date_count = int(invalid_dates.sum())
    if invalid_date_count:
        data = data.loc[~invalid_dates].copy()
        _add_issue(
            report,
            "game_date_validity",
            "high",
            f"{invalid_date_count:,} rows with invalid game dates removed.",
            "Games with invalid dates cannot support chronological validation.",
            "Re-pull or repair the affected source rows.",
        )

    duplicate_pitch_keys = data.duplicated(PITCH_KEY, keep="first")
    duplicate_pitch_count = int(duplicate_pitch_keys.sum())
    if duplicate_pitch_count:
        data = data.loc[~duplicate_pitch_keys].copy()
        _add_issue(
            report,
            "pitch_key_uniqueness",
            "high",
            f"{duplicate_pitch_count:,} duplicate pitch keys removed.",
            "Duplicate keys inflate pitch counts and distort within-game order.",
            "Audit source pagination and caching before production use.",
        )

    range_rules = {
        "release_speed": (50.0, 110.0, "mph"),
        "release_spin_rate": (0.0, 4_000.0, "rpm"),
        "release_extension": (3.0, 9.0, "feet"),
        "pfx_x": (-5.0, 5.0, "feet"),
        "pfx_z": (-5.0, 5.0, "feet"),
    }
    for column, (lower, upper, unit) in range_rules.items():
        if column not in data:
            continue
        invalid = data[column].notna() & ~data[column].between(lower, upper)
        count = int(invalid.sum())
        if count:
            data.loc[invalid, column] = np.nan
            rate = count / max(len(data), 1)
            _add_issue(
                report,
                f"{column}_range",
                "medium" if rate >= 0.01 else "low",
                f"{count:,} values ({rate:.2%}) outside {lower:g}-{upper:g} {unit} set to missing.",
                "Out-of-range tracking values would bias baselines and slopes.",
                "Monitor the rate and investigate if it exceeds 1%.",
            )

    if "game_type" in data:
        regular = data["game_type"].astype(str).eq("R")
        non_regular_count = int((~regular).sum())
        if non_regular_count:
            data = data.loc[regular].copy()
            report.metrics["non_regular_rows_removed"] = non_regular_count
    else:
        _add_issue(
            report,
            "game_type",
            "medium",
            "game_type is unavailable; regular-season filtering was not possible.",
            "Spring-training or postseason appearances could enter the analysis.",
            "Use a Statcast extract that includes game_type.",
        )

    data = data.sort_values(
        ["game_date", "game_pk", "at_bat_number", "pitch_number"],
        kind="mergesort",
    ).reset_index(drop=True)

    report.rows_clean = int(len(data))
    report.games = int(data["game_pk"].nunique())
    report.source_as_of = (
        data["game_date"].max().date().isoformat() if not data.empty else None
    )

    game_profile = data.groupby("game_pk", observed=True).agg(
        first_inning=("inning", "min"),
        pitch_rows=("game_pk", "size"),
    )
    eligible = game_profile[
        game_profile["first_inning"].eq(1)
        & game_profile["pitch_rows"].ge(config.min_pitches_per_start)
    ]
    report.eligible_starts = int(len(eligible))
    report.metrics.update(
        {
            "exact_duplicates_removed": exact_duplicates,
            "duplicate_pitch_keys_removed": duplicate_pitch_count,
            "eligible_start_rate": (
                report.eligible_starts / report.games if report.games else 0.0
            ),
            "release_speed_null_rate": float(data["release_speed"].isna().mean()),
            "pitch_type_null_rate": float(data["pitch_type"].isna().mean()),
            **{
                f"{label}_coverage": value
                for label, value in field_coverage(data).items()
            },
        }
    )

    if report.eligible_starts < config.min_starts:
        _add_issue(
            report,
            "starter_sample_size",
            "high",
            f"{report.eligible_starts} eligible starts; {config.min_starts} required for a stable season profile.",
            "Thresholds and pitch-type slopes may be unstable or unavailable.",
            "Use a fuller season, combine explicitly comparable seasons, or report insufficient data.",
        )

    if data.empty:
        _add_issue(
            report,
            "usable_rows",
            "critical",
            "No usable regular-season pitch rows remain after validation.",
            "No analysis can be computed.",
            "Re-pull the source data and inspect the filters.",
        )

    return data, report

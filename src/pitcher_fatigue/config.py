"""Central configuration for the pitcher-fatigue analysis."""

from __future__ import annotations

from dataclasses import dataclass


VALID_PITCH_TYPES = frozenset(
    {"FF", "SI", "FC", "SL", "ST", "SV", "CH", "CU", "KC", "FS", "FO", "CS"}
)
FASTBALL_TYPES = frozenset({"FF", "SI", "FC"})
BREAKING_TYPES = frozenset({"SL", "ST", "SV", "CU", "KC", "CS"})
OFFSPEED_TYPES = frozenset({"CH", "FS", "FO"})

PITCH_TYPE_NAMES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CH": "Changeup",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "FS": "Splitter",
    "FO": "Forkball",
    "CS": "Slow Curve",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Methodological choices shared across the pipeline.

    The defaults prioritize repeatability and conservative interpretation over
    producing a threshold for every pitcher.
    """

    baseline_window: int = 25
    baseline_min_pitches_per_type: int = 2
    min_pitches_per_start: int = 40
    min_starts: int = 10
    max_inning: int = 9
    pitch_bucket_size: int = 10
    max_pitch_count: int = 120
    fatigue_drop_mph: float = 1.0
    threshold_min_games: int = 5
    threshold_min_coverage: float = 0.30
    threshold_consecutive_buckets: int = 2
    field_coverage_warning_threshold: float = 0.80
    mnar_min_window_pitches: int = 50
    mnar_warning_threshold: float = 0.05
    asi_min_velocity_coverage: float = 0.90
    bootstrap_iterations: int = 500
    bootstrap_confidence: float = 0.95
    model_test_size: float = 0.20
    random_state: int = 42

    def validate(self) -> None:
        """Raise ``ValueError`` when configuration values are incoherent."""

        if self.baseline_window < 5:
            raise ValueError("baseline_window must be at least 5 pitches")
        if self.baseline_min_pitches_per_type < 1:
            raise ValueError("baseline_min_pitches_per_type must be positive")
        if self.min_pitches_per_start < self.baseline_window:
            raise ValueError("min_pitches_per_start must cover the baseline window")
        if not 0 < self.threshold_min_coverage <= 1:
            raise ValueError("threshold_min_coverage must be in (0, 1]")
        if self.threshold_consecutive_buckets < 1:
            raise ValueError("threshold_consecutive_buckets must be positive")
        if not 0 < self.field_coverage_warning_threshold <= 1:
            raise ValueError("field_coverage_warning_threshold must be in (0, 1]")
        if self.mnar_min_window_pitches < 1:
            raise ValueError("mnar_min_window_pitches must be positive")
        if not 0 <= self.mnar_warning_threshold <= 1:
            raise ValueError("mnar_warning_threshold must be in [0, 1]")
        if not 0 < self.asi_min_velocity_coverage <= 1:
            raise ValueError("asi_min_velocity_coverage must be in (0, 1]")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if not 0 < self.model_test_size < 0.5:
            raise ValueError("model_test_size must be in (0, 0.5)")

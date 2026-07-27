"""Shared labels and visual tokens for application and exported figures."""

from __future__ import annotations

import warnings


RED_SOX_RED = "#BD3039"
RED_SOX_NAVY = "#0C2340"
WHITE = "#FFFFFF"
SURFACE_SUBTLE = "#F7F8FA"
GRID_LIGHT = "#E8ECF0"
STROKE_LIGHT = "#DDE3E9"
TEXT_DARK = "#1A1A2E"
TEXT_MUTED = "#6B7280"
BLUE_LINE = "#2563EB"
REFERENCE_GOLD = "#B7791F"

# Streamlit ships Source Sans with the application, so the interface does not
# depend on a third-party font request. Plotly accepts the browser font stack;
# Matplotlib uses its bundled DejaVu Sans fallback for reproducible exports.
UI_FONT_FAMILY = '"Source Sans", "Source Sans Pro", Arial, sans-serif'
CHART_FONT_FAMILY = "Source Sans, Source Sans Pro, Arial, sans-serif"
EXPORT_FONT_FAMILY = "DejaVu Sans"

# Adapted from Thomas Nestico's public pitching_summary palette. SV and FO are
# explicit project extensions so every supported Statcast pitch code is styled.
PITCH_TYPE_COLORS = {
    "FF": "#FF007D",
    "SI": "#98165D",
    "FC": "#BE5FA0",
    "CH": "#F79E70",
    "FS": "#FE6100",
    "FO": "#D97706",
    "SL": "#67E18D",
    "ST": "#1BB999",
    "SV": "#376748",
    "KC": "#311D8B",
    "CU": "#3025CE",
    "CS": "#274BFC",
    "KN": "#867A08",
    "UN": "#9C8975",
}

DISPLAY_NAMES = {
    "speed_delta": "Velocity Delta (mph)",
    "release_speed": "Velocity (mph)",
    "effective_speed": "Perceived Velocity (mph)",
    "spin_delta_rpm": "Spin Rate Delta (rpm)",
    "release_spin_rate": "Spin Rate (rpm)",
    "pfx_x_delta_in": "Horizontal Break Delta (in)",
    "pfx_z_delta_in": "Induced Vertical Break Delta (in)",
    "pfx_x": "Horizontal Break (ft)",
    "pfx_z": "Induced Vertical Break (ft)",
    "movement_magnitude_in": "Total Movement (in)",
    "movement_delta_in": "Total Movement Delta (in)",
    "extension_delta_ft": "Extension Delta (ft)",
    "release_extension": "Extension (ft)",
    "game_pitch_count": "Game Pitch Count",
    "inning": "Inning",
    "pitch_type": "Pitch Type",
    "pitch_type_code": "Pitch Type",
    "pitch_type_name": "Pitch Type",
    "stand": "Batter Side",
    "is_fastball": "Fastball Family",
    "n_thruorder_pitcher": "Times Through Order",
    "times_through_order": "Times Through Order",
    "pitcher_days_since_prev_game": "Days Rest",
    "days_since_prev_game": "Days Rest",
    "spin_axis": "Spin Axis (°)",
}


def get_display_name(column: str) -> str:
    """Return an explicit reader-facing field label.

    Unknown fields remain visibly raw and emit a warning so new model features
    cannot silently ship with misleading units or jargon.
    """

    if column in DISPLAY_NAMES:
        return DISPLAY_NAMES[column]
    warnings.warn(
        f"No display name for column {column!r}; using the raw field name. "
        "Add it to DISPLAY_NAMES in presentation.py.",
        stacklevel=2,
    )
    return column

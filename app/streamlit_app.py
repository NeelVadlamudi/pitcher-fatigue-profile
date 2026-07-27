"""Interactive pitcher fatigue decision-support dashboard."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pitcher_fatigue.config import FASTBALL_TYPES, AnalysisConfig  # noqa: E402
from pitcher_fatigue.data_pull import DataPullError, load_or_pull  # noqa: E402
from pitcher_fatigue.features import aggregate_decay_curve  # noqa: E402
from pitcher_fatigue.outputs import summary_sheet_png_bytes  # noqa: E402
from pitcher_fatigue.pipeline import AnalysisBundle, analyze_pitcher_frame  # noqa: E402
from pitcher_fatigue.presentation import (  # noqa: E402
    BLUE_LINE,
    CHART_FONT_FAMILY,
    GRID_LIGHT,
    PITCH_TYPE_COLORS,
    RED_SOX_NAVY,
    RED_SOX_RED,
    REFERENCE_GOLD,
    STROKE_LIGHT,
    SURFACE_SUBTLE,
    TEXT_DARK,
    TEXT_MUTED,
    UI_FONT_FAMILY,
    get_display_name,
)
from pitcher_fatigue.sample_data import make_synthetic_pitcher  # noqa: E402


st.set_page_config(
    page_title="Pitcher Fatigue Profile",
    page_icon="⚾",
    layout="wide",
)

NAVY = RED_SOX_NAVY
BLUE = BLUE_LINE
RED = RED_SOX_RED
GOLD = REFERENCE_GOLD
MUTED = TEXT_MUTED
GRID = GRID_LIGHT
STROKE = STROKE_LIGHT
SURFACE = SURFACE_SUBTLE
FONT_STACK = UI_FONT_FAMILY


def apply_page_styles() -> None:
    """Apply a restrained typography and spacing system to the app shell."""

    st.markdown(
        f"""
<style>
:root {{
    --pf-navy: {NAVY};
    --pf-red: {RED};
    --pf-ink: {TEXT_DARK};
    --pf-muted: {MUTED};
    --pf-stroke: {STROKE};
    --pf-surface: {SURFACE};
}}

html, body, [data-testid="stAppViewContainer"] {{
    font-family: {FONT_STACK};
}}

[data-testid="stAppViewContainer"] {{
    background: #FCFCFB;
    color: var(--pf-ink);
}}

[data-testid="stHeader"] {{
    background: rgba(252, 252, 251, 0.94);
}}

[data-testid="stMainBlockContainer"] {{
    max-width: 1180px;
    padding-top: 2.25rem;
    padding-bottom: 3.5rem;
}}

[data-testid="stSidebar"] {{
    background: #F5F6F7;
    border-right: 1px solid var(--pf-stroke);
}}

[data-testid="stSidebarContent"] {{
    padding-top: 1.75rem;
}}

h1, h2, h3, h4 {{
    color: var(--pf-navy);
    font-family: {FONT_STACK};
    letter-spacing: -0.018em;
}}

h1 {{
    font-size: clamp(2.05rem, 4vw, 3rem);
    font-weight: 680;
    line-height: 1.02;
}}

h2 {{
    font-size: 1.42rem;
    font-weight: 660;
    line-height: 1.2;
}}

h3 {{
    font-size: 1.06rem;
    font-weight: 650;
    line-height: 1.3;
}}

p, li, label, [data-testid="stCaptionContainer"] {{
    line-height: 1.52;
}}

.pf-hero {{
    border-top: 4px solid var(--pf-red);
    padding: 1.25rem 0 1.35rem;
    margin-bottom: 0.35rem;
}}

.pf-eyebrow {{
    color: var(--pf-red);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    line-height: 1;
    margin: 0 0 0.72rem;
    text-transform: uppercase;
}}

.pf-hero h1 {{
    margin: 0;
}}

.pf-deck {{
    color: var(--pf-muted);
    font-size: 1.05rem;
    line-height: 1.52;
    margin: 0.72rem 0 0;
    max-width: 760px;
}}

.pf-section {{
    align-items: baseline;
    border-bottom: 1px solid var(--pf-stroke);
    display: flex;
    gap: 1rem;
    justify-content: space-between;
    margin: 1.65rem 0 1rem;
    padding-bottom: 0.58rem;
}}

.pf-section h2 {{
    font-size: 1.18rem;
    margin: 0;
}}

.pf-section p {{
    color: var(--pf-muted);
    font-size: 0.86rem;
    margin: 0;
    text-align: right;
}}

.pf-sidebar-mark {{
    color: var(--pf-red);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    margin: 0 0 0.35rem;
    text-transform: uppercase;
}}

.pf-sidebar-title {{
    color: var(--pf-navy);
    font-size: 1.32rem;
    font-weight: 680;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0 0 0.35rem;
}}

.pf-sidebar-copy {{
    color: var(--pf-muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0 0 1.35rem;
}}

[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid var(--pf-stroke);
    border-radius: 7px;
    min-height: 108px;
    padding: 0.88rem 1rem 0.78rem;
}}

[data-testid="stMetricLabel"] {{
    color: var(--pf-muted);
    font-size: 0.76rem;
    font-weight: 650;
    letter-spacing: 0.025em;
}}

[data-testid="stMetricValue"] {{
    color: var(--pf-navy);
    font-size: 1.52rem;
    font-weight: 670;
    letter-spacing: -0.025em;
    line-height: 1.15;
}}

[data-baseweb="tab-list"] {{
    border-bottom: 1px solid var(--pf-stroke);
    gap: 1.35rem;
}}

[data-baseweb="tab"] {{
    color: var(--pf-muted);
    font-size: 0.88rem;
    font-weight: 620;
    padding-left: 0;
    padding-right: 0;
}}

[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--pf-navy);
}}

[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border: 1px solid var(--pf-stroke);
    border-radius: 7px;
    padding: 0.2rem;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid var(--pf-stroke);
    border-radius: 7px;
    overflow: hidden;
}}

.stButton > button,
.stDownloadButton > button {{
    border-radius: 5px;
    font-weight: 650;
    min-height: 2.6rem;
}}

div[data-baseweb="notification"] {{
    border-radius: 6px;
}}

hr {{
    border-color: var(--pf-stroke);
    margin: 2rem 0 1.25rem;
}}

@media (max-width: 760px) {{
    [data-testid="stMainBlockContainer"] {{
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.35rem;
    }}
    .pf-section {{
        align-items: flex-start;
        flex-direction: column;
        gap: 0.22rem;
    }}
    .pf-section p {{
        text-align: left;
    }}
}}

@media (max-width: 1050px) {{
    [data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap;
    }}
    [data-testid="stMainBlockContainer"]
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        flex: 1 1 calc(50% - 0.5rem) !important;
        min-width: calc(50% - 0.5rem) !important;
        width: calc(50% - 0.5rem) !important;
    }}
}}

@media (max-width: 620px) {{
    [data-testid="stMainBlockContainer"]
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        flex-basis: 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


def section_heading(title: str, detail: str) -> None:
    """Render a consistent section title and supporting line."""

    st.markdown(
        f"""
<div class="pf-section">
  <h2>{title}</h2>
  <p>{detail}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def apply_chart_style(
    figure: go.Figure,
    *,
    height: int,
    margin: dict[str, int],
    showlegend: bool | None = None,
) -> go.Figure:
    """Apply the shared chart typography, grid, and spacing contract."""

    updates: dict[str, object] = {
        "height": height,
        "margin": margin,
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "font": {
            "family": CHART_FONT_FAMILY,
            "color": TEXT_DARK,
            "size": 13,
        },
        "title": {
            "x": 0.015,
            "xanchor": "left",
            "font": {
                "family": CHART_FONT_FAMILY,
                "color": NAVY,
                "size": 19,
            },
        },
        "hoverlabel": {
            "font": {"family": CHART_FONT_FAMILY, "size": 13},
        },
    }
    if showlegend is not None:
        updates["showlegend"] = showlegend
    figure.update_layout(**updates)
    figure.update_xaxes(
        gridcolor=GRID,
        linecolor=STROKE,
        tickcolor=STROKE,
        tickfont={"color": MUTED, "size": 12},
        title_font={"color": MUTED, "size": 12},
        title_standoff=14,
        zeroline=False,
    )
    figure.update_yaxes(
        gridcolor=GRID,
        linecolor=STROKE,
        tickcolor=STROKE,
        tickfont={"color": MUTED, "size": 12},
        title_font={"color": MUTED, "size": 12},
        title_standoff=14,
        zeroline=False,
    )
    return figure


apply_page_styles()


@st.cache_data(show_spinner=False)
def load_sample_data() -> tuple[pd.DataFrame, dict[str, object]]:
    path = ROOT / "data" / "sample" / "sample_pitcher.csv"
    if path.exists():
        frame = pd.read_csv(path, low_memory=False)
    else:
        frame = make_synthetic_pitcher()
    return frame, {
        "pitcher_name": "Synthetic Demonstration Pitcher",
        "season": 2024,
        "source": "deterministic_synthetic_demo",
        "cache_path": str(path),
        "rows": int(len(frame)),
    }


@st.cache_data(ttl=86_400, show_spinner=False)
def load_live_data(
    first_name: str,
    last_name: str,
    season: int,
    force_refresh: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return load_or_pull(
        first_name,
        last_name,
        season,
        cache_dir=ROOT / "data/raw",
        force_refresh=force_refresh,
    )


@st.cache_resource(show_spinner=False)
def run_analysis(
    frame: pd.DataFrame,
    provenance_json: str,
) -> AnalysisBundle:
    return analyze_pitcher_frame(
        frame,
        provenance=json.loads(provenance_json),
        config=AnalysisConfig(),
        train_models=True,
        model_backend="auto",
    )


def threshold_display(bundle: AnalysisBundle) -> tuple[str, str]:
    threshold = bundle.threshold
    if threshold.status == "established":
        return f"Pitches {threshold.threshold_range}", "Supported sustained decline"
    if threshold.status == "not_reached":
        return "Not established", "No sustained supported 1 mph decline"
    return "Insufficient data", "Late-game coverage below the reliability bar"


def velocity_curve_figure(
    curve: pd.DataFrame,
    bundle: AnalysisBundle,
    *,
    title: str,
    subtitle: str,
    y_title: str,
    reference_value: float | None = None,
    show_threshold: bool = False,
) -> go.Figure:
    figure = go.Figure()
    if not curve.empty:
        x = curve["pitch_count_bucket_mid"]
        figure.add_trace(
            go.Scatter(
                x=pd.concat([x, x.iloc[::-1]]),
                y=pd.concat([curve["ci_upper"], curve["ci_lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(47,102,144,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                hoverinfo="skip",
                name="95% game-cluster interval",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=x,
                y=curve["mean_delta"],
                mode="lines+markers",
                line={"color": BLUE, "width": 3},
                marker={"size": 7, "color": BLUE},
                customdata=curve[["pitch_count_bucket", "games", "pitches", "coverage"]],
                hovertemplate=(
                    "Pitches %{customdata[0]}<br>"
                    "Mean delta: %{y:.2f}<br>"
                    "Games: %{customdata[1]}<br>"
                    "Pitches: %{customdata[2]}<br>"
                    "Start coverage: %{customdata[3]:.0%}<extra></extra>"
                ),
                name="Equal-game mean",
            )
        )
    figure.add_hline(y=0, line={"color": MUTED, "dash": "dash", "width": 1})
    if reference_value is not None:
        figure.add_hline(
            y=reference_value,
            line={"color": GOLD, "dash": "dot", "width": 1.5},
            annotation_text=f"{reference_value:g} reference",
            annotation_position="bottom right",
        )
    if show_threshold and bundle.threshold.status == "established":
        start = int(bundle.threshold.threshold_pitch or 0)
        figure.add_vrect(
            x0=start,
            x1=start + bundle.config.pitch_bucket_size,
            fillcolor=GOLD,
            opacity=0.10,
            line_width=0,
            annotation_text=f"Threshold {bundle.threshold.threshold_range}",
            annotation_position="top left",
        )
    figure.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>"},
        xaxis_title="Game pitch count",
        yaxis_title=y_title,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.02,
            "x": 0,
            "font": {"size": 11, "color": MUTED},
        },
    )
    return apply_chart_style(
        figure,
        height=440,
        margin={"l": 62, "r": 28, "t": 98, "b": 58},
    )


def slope_figure(slopes: pd.DataFrame) -> go.Figure:
    eligible = slopes[slopes["eligible"]].sort_values(
        "slope_mph_per_10", ascending=True
    )
    figure = go.Figure()
    if not eligible.empty:
        lower = eligible["slope_mph_per_10"] - eligible["ci_lower"]
        upper = eligible["ci_upper"] - eligible["slope_mph_per_10"]
        figure.add_trace(
            go.Bar(
                x=eligible["slope_mph_per_10"],
                y=eligible["pitch_type_name"],
                orientation="h",
                marker_color=[
                    PITCH_TYPE_COLORS.get(code, MUTED)
                    for code in eligible["pitch_type"]
                ],
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": upper,
                    "arrayminus": lower,
                    "color": NAVY,
                    "thickness": 1,
                },
                customdata=eligible[["games", "pitches", "late_delta"]],
                hovertemplate=(
                    "%{y}<br>Slope: %{x:.3f} mph / 10 pitches<br>"
                    "Games: %{customdata[0]}<br>Pitches: %{customdata[1]}<br>"
                    "Mean delta at pitch 80+: %{customdata[2]:.2f} mph"
                    "<extra></extra>"
                ),
            )
        )
    figure.add_vline(x=0, line={"color": MUTED, "width": 1})
    figure.update_layout(
        title={
            "text": (
                "Velocity slope by pitch type"
                "<br><sup>Mean of per-game slopes; 95% game bootstrap intervals</sup>"
            )
        },
        xaxis_title="Velocity change per 10 game pitches (mph)",
        yaxis_title="",
    )
    return apply_chart_style(
        figure,
        height=390,
        margin={"l": 118, "r": 30, "t": 88, "b": 58},
        showlegend=False,
    )


def importance_figure(bundle: AnalysisBundle) -> go.Figure:
    importance = bundle.feature_importance.head(10).sort_values("importance")
    labels = [
        get_display_name(str(value)) for value in importance["feature"]
    ]
    method = (bundle.importance_method or "unavailable").replace("_", " ")
    figure = go.Figure(
        go.Bar(
            x=importance["importance"],
            y=labels,
            orientation="h",
            marker_color=NAVY,
            customdata=importance[["share"]],
            hovertemplate=(
                "%{y}<br>Importance: %{x:.4f}<br>"
                "Share: %{customdata[0]:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title={
            "text": (
                "Associative model feature importance"
                f"<br><sup>{method}; importance is not a causal effect</sup>"
            )
        },
        xaxis_title="Contribution to model accuracy",
        yaxis_title="",
    )
    return apply_chart_style(
        figure,
        height=410,
        margin={"l": 178, "r": 28, "t": 88, "b": 58},
        showlegend=False,
    )


def prediction_figure(bundle: AnalysisBundle) -> go.Figure:
    prediction = bundle.pre_pitch_model.prediction_frame()
    lower = float(
        min(
            prediction["actual_speed_delta"].min(),
            prediction["predicted_speed_delta"].min(),
        )
    )
    upper = float(
        max(
            prediction["actual_speed_delta"].max(),
            prediction["predicted_speed_delta"].max(),
        )
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=prediction["actual_speed_delta"],
            y=prediction["predicted_speed_delta"],
            mode="markers",
            marker={"color": BLUE, "size": 6, "opacity": 0.55},
            customdata=prediction[["game_pk", "absolute_error"]],
            hovertemplate=(
                "Game %{customdata[0]}<br>Actual: %{x:.2f} mph<br>"
                "Predicted: %{y:.2f} mph<br>Absolute error: "
                "%{customdata[1]:.2f} mph<extra></extra>"
            ),
            name="Held-out pitch",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line={"color": MUTED, "dash": "dash"},
            hoverinfo="skip",
            name="Perfect prediction",
        )
    )
    figure.update_layout(
        title={
            "text": (
                "Held-out-game predictions"
                "<br><sup>Every test-game pitch is excluded from model training</sup>"
            )
        },
        xaxis_title="Observed velocity delta (mph)",
        yaxis_title="Predicted velocity delta (mph)",
    )
    apply_chart_style(
        figure,
        height=410,
        margin={"l": 64, "r": 28, "t": 88, "b": 60},
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


st.markdown(
    """
<header class="pf-hero">
  <p class="pf-eyebrow">Statcast decision support</p>
  <h1>Pitcher Fatigue Profile</h1>
  <p class="pf-deck">
    A game-by-game view of how a starter's velocity and pitch quality change as
    pitch count rises. Built for review after the game—not live prediction or
    injury assessment.
  </p>
</header>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
<p class="pf-sidebar-mark">Pitcher workload</p>
<p class="pf-sidebar-title">Build a profile</p>
<p class="pf-sidebar-copy">
Choose a data source and pitcher-season. The analysis keeps entire games
together when it evaluates the model.
</p>
""",
        unsafe_allow_html=True,
    )
    source_choice = st.radio(
        "Data source",
        ["Bundled synthetic demo", "Live Statcast"],
        help="The demo is deterministic and does not represent a real player.",
    )

    requested_run = source_choice == "Bundled synthetic demo"
    if source_choice == "Live Statcast":
        first_name = st.text_input("First name", value="Logan")
        last_name = st.text_input("Last name", value="Webb")
        seasons = list(range(date.today().year, 2014, -1))
        season = st.selectbox("Season", seasons, index=min(1, len(seasons) - 1))
        force_refresh = st.checkbox(
            "Refresh cached Statcast data",
            value=False,
            help="Fetch the pitcher-season again instead of using the local CSV cache.",
        )
        if season == date.today().year:
            st.caption("The current season is partial as of today.")
        requested_run = st.button("Run analysis", type="primary", width="stretch")
    else:
        first_name = "Synthetic"
        last_name = "Demonstration Pitcher"
        season = 2024
        force_refresh = False
        st.info(
            "The demo contains a known late-game decline and runs without an "
            "internet connection."
        )

if not requested_run:
    st.info("Choose a pitcher and select **Run analysis**.")
    st.stop()

try:
    if source_choice == "Bundled synthetic demo":
        raw, provenance = load_sample_data()
    else:
        with st.spinner("Loading pitch-level Statcast data…"):
            raw, provenance = load_live_data(
                first_name,
                last_name,
                int(season),
                force_refresh,
            )
except (DataPullError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(
        "Statcast data could not be loaded. Baseball Savant may be temporarily "
        f"unavailable. Technical detail: {exc}"
    )
    st.stop()

if raw.empty or len(raw) < 100:
    st.error(
        "No usable pitch sample came back for this pitcher and season. "
        "Check the name and season and try again. If the problem continues, "
        "Baseball Savant may be temporarily unavailable."
    )
    st.stop()

try:
    with st.spinner("Validating data and fitting game-isolated models…"):
        bundle = run_analysis(raw, json.dumps(provenance, sort_keys=True))
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.stop()

pitcher_name = str(provenance.get("pitcher_name", f"{first_name} {last_name}"))
is_synthetic = provenance.get("source") == "deterministic_synthetic_demo"
if is_synthetic:
    st.warning(
        "Synthetic demonstration results — do not interpret these values as an MLB finding.",
        icon="⚠️",
    )

threshold_value, threshold_help = threshold_display(bundle)
asi = bundle.arm_stamina_index.score
mae = (
    bundle.pre_pitch_model.metrics["mae"]
    if bundle.pre_pitch_model is not None
    else None
)

section_heading(
    "Season summary",
    f"{pitcher_name} · {int(season)} regular season",
)
metric_columns = st.columns(4, gap="small")
metric_columns[0].metric(
    "Fatigue threshold",
    threshold_value,
    help=threshold_help,
)
metric_columns[1].metric(
    "Experimental ASI",
    f"{asi:.1f} / 100" if asi is not None else "Unavailable",
    help=bundle.arm_stamina_index.caveat,
)
metric_columns[2].metric(
    "Eligible starts",
    f"{bundle.coverage['games']}",
    help=f"Traditional starts with at least {bundle.config.min_pitches_per_start} observed pitches.",
)
metric_columns[3].metric(
    "Held-out MAE",
    f"{mae:.2f} mph" if mae is not None else "Unavailable",
    help="Pre-pitch model error on entire chronologically held-out games.",
)

source_label = str(provenance.get("source", "unknown"))
if source_label == "local_cache":
    source_label = "local cache of Baseball Savant via pybaseball"
st.caption(
    f"Source: {source_label} · Data as of "
    f"{bundle.quality_report.source_as_of or 'unknown'} · Quality status: "
    f"{bundle.quality_report.status.replace('_', ' ')}"
)
if source_choice == "Live Statcast" and provenance.get("cached_at"):
    cache_action = (
        "This run refreshed the source cache."
        if force_refresh
        else "Enable “Refresh cached Statcast data” and rerun to fetch current data."
    )
    st.caption(
        f"Local cache timestamp: {provenance['cached_at']}. "
        "Statcast pitch classifications can be revised after a game. "
        f"{cache_action}"
    )

spin_coverage = float(bundle.coverage.get("spin_rate_coverage", 0.0) or 0.0)
if spin_coverage < bundle.config.field_coverage_warning_threshold:
    st.warning(
        f"Spin rate is tracked on {spin_coverage:.0%} of eligible pitch rows. "
        "Spin-retention results may be unreliable; velocity results are unaffected."
    )
spin_mnar = bundle.coverage.get("spin_rate_mnar_shift")
if (
    spin_mnar is not None
    and float(spin_mnar) > bundle.config.mnar_warning_threshold
):
    st.warning(
        f"Spin rate is missing {float(spin_mnar):.0%} more often after pitch 80 "
        "than during pitches 1–30. Late-game spin estimates may therefore be "
        "less representative; the diagnostic does not establish why values are missing."
    )

section_heading(
    "Evidence",
    "Start with the game-level trend, then inspect pitch quality and model fit.",
)
overview_tab, quality_tab, model_tab, methods_tab = st.tabs(
    ["Overview", "Pitch quality", "Model validation", "Data & methods"]
)

with overview_tab:
    st.plotly_chart(
        velocity_curve_figure(
            bundle.velocity_curve,
            bundle,
            title="Fastball velocity delta by game pitch count",
            subtitle=(
                "Same-pitch-type early baseline; equal game weights;<br>"
                "95% game-cluster intervals"
            ),
            y_title="Velocity delta (mph)",
            reference_value=-bundle.config.fatigue_drop_mph,
            show_threshold=True,
        ),
        width="stretch",
    )
    st.plotly_chart(slope_figure(bundle.pitch_type_slopes), width="stretch")

    eligible = bundle.pitch_type_slopes[bundle.pitch_type_slopes["eligible"]]
    if not eligible.empty:
        fastest = eligible.sort_values("slope_mph_per_10").iloc[0]
        st.info(
            f"Most negative supported pitch-type slope: **{fastest['pitch_type_name']}** "
            f"at {fastest['slope_mph_per_10']:.2f} mph per 10 game pitches. "
            "This is a descriptive association, not evidence that pitch count alone caused the change."
        )

with quality_tab:
    measure_column, pitch_column = st.columns([1, 2], gap="medium")
    with measure_column:
        metric_choice = st.selectbox(
            "Pitch-quality measure",
            ["Spin rate", "Movement magnitude", "Velocity"],
        )
    with pitch_column:
        selected_pitch_types = st.multiselect(
            "Pitch types",
            options=sorted(bundle.features["pitch_type"].dropna().unique()),
            default=sorted(
                set(bundle.features["pitch_type"].dropna().unique())
                & set(FASTBALL_TYPES)
            ),
        )
    if not selected_pitch_types:
        st.warning("Select at least one pitch type.")
    else:
        if metric_choice == "Spin rate":
            metric_name = "spin_delta_rpm"
            y_title = "Spin-rate delta (rpm)"
            reference = None
        elif metric_choice == "Movement magnitude":
            metric_name = "movement_delta_in"
            y_title = "Movement-magnitude delta (inches)"
            reference = None
        else:
            metric_name = "speed_delta"
            y_title = "Velocity delta (mph)"
            reference = -bundle.config.fatigue_drop_mph
        if metric_name in bundle.features:
            selected_curve = aggregate_decay_curve(
                bundle.features,
                pitch_types=selected_pitch_types,
                metric=metric_name,
                config=bundle.config,
            )
            st.plotly_chart(
                velocity_curve_figure(
                    selected_curve,
                    bundle,
                    title=f"{metric_choice} delta by game pitch count",
                    subtitle=(
                        f"Pitch types: {', '.join(selected_pitch_types)};<br>"
                        "same-type early baseline"
                    ),
                    y_title=y_title,
                    reference_value=reference,
                ),
                width="stretch",
            )
        else:
            st.warning(f"{metric_choice} is unavailable in this data source.")

    display_columns = [
        "pitch_type_name",
        "slope_mph_per_10",
        "ci_lower",
        "ci_upper",
        "games",
        "pitches",
        "late_delta",
        "eligible",
    ]
    st.dataframe(
        bundle.pitch_type_slopes[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "pitch_type_name": "Pitch type",
            "slope_mph_per_10": st.column_config.NumberColumn(
                "Velocity slope / 10 pitches", format="%.3f"
            ),
            "ci_lower": st.column_config.NumberColumn("95% CI lower", format="%.3f"),
            "ci_upper": st.column_config.NumberColumn("95% CI upper", format="%.3f"),
            "late_delta": st.column_config.NumberColumn(
                "Mean delta at pitch 80+", format="%.2f"
            ),
        },
    )

with model_tab:
    if bundle.pre_pitch_model is None:
        st.warning(bundle.model_error or "The modeling sample is insufficient.")
    else:
        metrics = bundle.pre_pitch_model.metrics
        model_columns = st.columns(4, gap="small")
        model_columns[0].metric("MAE", f"{metrics['mae']:.3f} mph")
        model_columns[1].metric("RMSE", f"{metrics['rmse']:.3f} mph")
        model_columns[2].metric("R²", f"{metrics['r2']:.3f}")
        model_columns[3].metric(
            "Naive-baseline improvement",
            f"{metrics['mae_improvement_pct']:.1f}%",
        )
        st.caption(
            f"Backend: {bundle.pre_pitch_model.backend} · "
            f"Train games: {metrics['train_games']} · Test games: "
            f"{metrics['test_games']} · Split: {metrics['split_strategy']}"
        )
        left, right = st.columns(2, gap="medium")
        with left:
            st.plotly_chart(prediction_figure(bundle), width="stretch")
        with right:
            if not bundle.feature_importance.empty:
                st.plotly_chart(importance_figure(bundle), width="stretch")
        st.info(
            "The prediction chart evaluates a pre-pitch context model. The feature-importance "
            "chart uses a separate descriptive-quality model that may include measurements "
            "from the same pitch. It explains model associations and is not a causal or "
            "real-time forecasting claim."
        )
        if float(metrics["r2"]) < 0.05:
            st.info(
                f"R² is {metrics['r2']:.3f}. Pre-pitch context explains little "
                "individual-pitch variation for this season. The decay curve answers "
                "a different question: it summarizes game-level averages with "
                "uncertainty rather than forecasting each pitch."
            )

with methods_tab:
    st.subheader("Data quality")
    issues = bundle.quality_report.to_frame()
    if issues.empty:
        st.success("No material automated data-quality issues were detected.")
    else:
        st.dataframe(issues, hide_index=True, width="stretch")

    coverage_columns = st.columns(4, gap="small")
    coverage_columns[0].metric(
        "Baseline coverage", f"{bundle.coverage['baseline_coverage']:.1%}"
    )
    coverage_columns[1].metric(
        "Late-game starts", f"{bundle.coverage['late_games_80_plus']}"
    )
    coverage_columns[2].metric(
        "Pitch types", f"{bundle.coverage['pitch_types']}"
    )
    coverage_columns[3].metric(
        "Clean pitch rows", f"{bundle.quality_report.rows_clean:,}"
    )

    st.subheader("Threshold interpretation")
    st.write(bundle.threshold.reason)
    st.caption(
        "A threshold requires a sustained average decline of at least "
        f"{bundle.config.fatigue_drop_mph:g} mph, a bootstrap upper interval below "
        "zero, at least "
        f"{bundle.config.threshold_min_games} games, and at least "
        f"{bundle.config.threshold_min_coverage:.0%} start coverage."
    )

    with st.expander("Methodological guardrails"):
        st.markdown(
            """
- Baselines are calculated separately for every game and pitch type.
- Complete games—not individual pitches—are assigned to train or test.
- The primary holdout is chronological: the latest games are unseen during training.
- Bucket estimates weight games equally and bootstrap games as clusters.
- Late buckets display contributing-game coverage to expose survivor bias.
- SHAP or permutation importance is associative, not causal.
- The Arm Stamina Index is experimental and not population-validated.
- This tool does not estimate injury risk or replace coaching judgment.
"""
        )

section_heading(
    "Export",
    "Share the current profile as an image or inspect the feature-level data.",
)
download_columns = st.columns([1, 1, 1], gap="small")
safe_name = re.sub(r"[^a-z0-9]+", "_", pitcher_name.lower()).strip("_")
with download_columns[0]:
    try:
        png = summary_sheet_png_bytes(
            bundle,
            pitcher_name,
            int(season),
            dpi=150,
        )
        st.download_button(
            "Summary PNG (web)",
            data=png,
            file_name=f"{safe_name}_{season}_fatigue_profile.png",
            mime="image/png",
            width="stretch",
        )
    except Exception as exc:
        st.caption(f"Summary image unavailable: {exc}")
with download_columns[1]:
    try:
        print_png = summary_sheet_png_bytes(
            bundle,
            pitcher_name,
            int(season),
            dpi=300,
        )
        st.download_button(
            "Summary PNG (print)",
            data=print_png,
            file_name=f"{safe_name}_{season}_fatigue_profile_hires.png",
            mime="image/png",
            width="stretch",
        )
    except Exception as exc:
        st.caption(f"Print image unavailable: {exc}")
with download_columns[2]:
    st.download_button(
        "Feature data (CSV)",
        data=bundle.features.to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_name}_{season}_features.csv",
        mime="text/csv",
        width="stretch",
    )
st.caption(
    "Downloads preserve the analysis snapshot shown above. Raw Statcast files "
    "remain local and are excluded from version control."
)

"""Publication-ready static outputs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .pipeline import AnalysisBundle
from .presentation import (
    BLUE_LINE,
    EXPORT_FONT_FAMILY,
    GRID_LIGHT,
    PITCH_TYPE_COLORS,
    RED_SOX_NAVY,
    RED_SOX_RED,
    REFERENCE_GOLD,
    TEXT_DARK,
    TEXT_MUTED,
    WHITE,
)


def _threshold_label(bundle: AnalysisBundle) -> str:
    if bundle.threshold.status == "established":
        return f"Pitches {bundle.threshold.threshold_range}"
    if bundle.threshold.status == "not_reached":
        return "Not established"
    return "Insufficient data"


def generate_summary_sheet(
    bundle: AnalysisBundle,
    pitcher_name: str,
    season: int,
    *,
    save_path: str | Path | None = None,
    dpi: int = 150,
):
    """Generate a coach-facing one-page pitcher fatigue summary sheet."""

    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    matplotlib.rcParams.update(
        {
            "font.family": EXPORT_FONT_FAMILY,
            "font.size": 9,
            "axes.titlecolor": TEXT_DARK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT_DARK,
            "axes.labelsize": 9,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
        }
    )
    fig = plt.figure(figsize=(14, 8), facecolor=WHITE, dpi=dpi)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.1, 0.8],
        hspace=0.34,
        wspace=0.34,
        left=0.06,
        right=0.97,
        top=0.79,
        bottom=0.12,
    )
    ax_curve = fig.add_subplot(grid[0, :2])
    ax_slopes = fig.add_subplot(grid[0, 2])
    ax_findings = fig.add_subplot(grid[1, :2])
    ax_metrics = fig.add_subplot(grid[1, 2])

    curve = bundle.velocity_curve
    if not curve.empty:
        x = curve["pitch_count_bucket_mid"].to_numpy(dtype=float)
        y = curve["mean_delta"].to_numpy(dtype=float)
        lower = curve["ci_lower"].to_numpy(dtype=float)
        upper = curve["ci_upper"].to_numpy(dtype=float)
        ax_curve.fill_between(
            x,
            lower,
            upper,
            color=BLUE_LINE,
            alpha=0.14,
            linewidth=0,
        )
        ax_curve.plot(
            x,
            y,
            color=BLUE_LINE,
            marker="o",
            markerfacecolor=WHITE,
            markeredgecolor=BLUE_LINE,
            linewidth=2.4,
            markersize=4.5,
        )
        last_two = curve.tail(2)
        if (
            len(last_two) == 2
            and last_two["mean_delta"].iloc[-1]
            > last_two["mean_delta"].iloc[-2]
            and last_two["coverage"].iloc[-1] < 0.40
        ):
            ax_curve.annotate(
                "Late uptick may reflect pitchers having\n"
                "better outings at this pitch count\n"
                "(see contributing-start coverage)",
                xy=(
                    last_two["pitch_count_bucket_mid"].iloc[-1],
                    last_two["mean_delta"].iloc[-1],
                ),
                xytext=(-145, 28),
                textcoords="offset points",
                fontsize=7,
                color=TEXT_MUTED,
                arrowprops={
                    "arrowstyle": "->",
                    "color": TEXT_MUTED,
                    "linewidth": 0.8,
                },
            )
    ax_curve.axhline(0, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax_curve.axhline(
        -bundle.config.fatigue_drop_mph,
        color=REFERENCE_GOLD,
        linewidth=1.2,
        linestyle=":",
        label=f"-{bundle.config.fatigue_drop_mph:g} mph reference",
    )
    if bundle.threshold.status == "established":
        start = float(bundle.threshold.threshold_pitch or 0)
        ax_curve.axvspan(
            start,
            start + bundle.config.pitch_bucket_size,
            color=REFERENCE_GOLD,
            alpha=0.10,
            label=f"Threshold {bundle.threshold.threshold_range}",
        )
    ax_curve.set_title(
        "Fastball velocity delta by game pitch count",
        loc="left",
        pad=8,
    )
    ax_curve.set_xlabel("Game pitch count (10-pitch buckets)")
    ax_curve.set_ylabel("Velocity delta from same-type baseline (mph)")
    ax_curve.legend(frameon=False, fontsize=8, loc="lower left")

    slopes = bundle.pitch_type_slopes
    eligible_slopes = (
        slopes[slopes["eligible"].eq(True)].copy()
        if "eligible" in slopes
        else slopes.iloc[0:0].copy()
    )
    if not eligible_slopes.empty:
        eligible_slopes = eligible_slopes.sort_values("slope_mph_per_10")
        positions = np.arange(len(eligible_slopes))
        values = eligible_slopes["slope_mph_per_10"].to_numpy(dtype=float)
        errors = np.vstack(
            [
                values - eligible_slopes["ci_lower"].to_numpy(dtype=float),
                eligible_slopes["ci_upper"].to_numpy(dtype=float) - values,
            ]
        )
        ax_slopes.barh(
            positions,
            values,
            color=[
                PITCH_TYPE_COLORS.get(code, TEXT_MUTED)
                for code in eligible_slopes["pitch_type"]
            ],
            alpha=0.90,
            xerr=errors,
            edgecolor=WHITE,
            linewidth=0.5,
            error_kw={
                "ecolor": RED_SOX_NAVY,
                "elinewidth": 1,
                "capsize": 2,
            },
        )
        ax_slopes.set_yticks(
            positions, labels=eligible_slopes["pitch_type_name"].tolist()
        )
        ax_slopes.axvline(0, color=TEXT_MUTED, linewidth=1)
    else:
        ax_slopes.text(
            0.5,
            0.5,
            "Insufficient pitch-type\nslope coverage",
            ha="center",
            va="center",
            transform=ax_slopes.transAxes,
            color=TEXT_MUTED,
        )
    ax_slopes.set_title(
        "Velocity slope by pitch type",
        loc="left",
        y=1.075,
        pad=0,
    )
    ax_slopes.text(
        0,
        1.02,
        "Mean of per-game slopes · 95% game intervals",
        transform=ax_slopes.transAxes,
        color=TEXT_MUTED,
        fontsize=7.5,
        ha="left",
        va="bottom",
    )
    ax_slopes.set_xlabel("mph per 10 game pitches")

    model_mae = (
        bundle.pre_pitch_model.metrics["mae"]
        if bundle.pre_pitch_model is not None
        else None
    )
    ax_findings.axis("off")
    ax_findings.set_title("Key findings", loc="left", pad=8)
    if not eligible_slopes.empty:
        most_declining = eligible_slopes.sort_values("slope_mph_per_10").iloc[0]
        pitch_finding = (
            f"{most_declining['pitch_type_name']}: "
            f"{most_declining['slope_mph_per_10']:.2f} mph per 10 pitches"
        )
    else:
        pitch_finding = "Insufficient pitch-type slope coverage"
    if bundle.threshold.status == "established":
        planning_note = (
            "This is the first sustained, coverage-qualified decline range. "
            "It supports review, not an automatic removal decision."
        )
    elif bundle.threshold.status == "not_reached":
        planning_note = (
            "No pitch-count window met every support rule. Use the curve and "
            "outing context rather than a hard cutoff."
        )
    else:
        planning_note = (
            "Late-game coverage is too limited to support a defensible hard cutoff."
        )
    findings = [
        ("Threshold", _threshold_label(bundle)),
        ("Most negative supported slope", pitch_finding),
        (
            "Same-type baseline coverage",
            f"{bundle.coverage['baseline_coverage']:.1%} of feature rows",
        ),
        (
            "Held-out model MAE",
            f"{model_mae:.2f} mph" if model_mae is not None else "Unavailable",
        ),
    ]
    y_position = 0.87
    for label, value in findings:
        ax_findings.text(
            0.0,
            y_position,
            f"{label}:",
            transform=ax_findings.transAxes,
            color=TEXT_MUTED,
            fontsize=8,
            fontweight="bold",
            va="top",
        )
        ax_findings.text(
            0.34,
            y_position,
            value,
            transform=ax_findings.transAxes,
            color=TEXT_DARK,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
        y_position -= 0.15
    ax_findings.text(
        0.0,
        0.14,
        fill(planning_note, width=88),
        transform=ax_findings.transAxes,
        color=TEXT_DARK,
        fontsize=9,
        va="top",
        linespacing=1.35,
    )

    ax_metrics.axis("off")
    ax_metrics.set_title("Season at a glance", loc="left", pad=8)
    asi = bundle.arm_stamina_index.score
    metrics = [
        ("FATIGUE THRESHOLD", _threshold_label(bundle)),
        ("EXPERIMENTAL ASI", f"{asi:.1f} / 100" if asi is not None else "Unavailable"),
        ("ELIGIBLE STARTS", f"{bundle.coverage['games']}"),
        ("LATE-GAME STARTS", f"{bundle.coverage['late_games_80_plus']}"),
        ("HOLDOUT MAE", f"{model_mae:.2f} mph" if model_mae is not None else "Unavailable"),
        ("DATA AS OF", bundle.quality_report.source_as_of or "Unknown"),
    ]
    y_position = 0.96
    for label, value in metrics:
        ax_metrics.text(
            0.02,
            y_position,
            label,
            transform=ax_metrics.transAxes,
            color=TEXT_MUTED,
            fontsize=8,
            fontweight="bold",
        )
        ax_metrics.text(
            0.02,
            y_position - 0.075,
            value,
            transform=ax_metrics.transAxes,
            color=TEXT_DARK,
            fontsize=13,
            fontweight="bold",
        )
        y_position -= 0.155

    ax_curve.grid(axis="y", color=GRID_LIGHT, linewidth=0.7)
    ax_slopes.grid(axis="x", color=GRID_LIGHT, linewidth=0.7)
    for axis in [ax_curve, ax_slopes]:
        axis.set_facecolor(WHITE)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID_LIGHT)
        axis.tick_params(colors=TEXT_MUTED, labelsize=8)
        axis.title.set_color(TEXT_DARK)
        axis.set_axisbelow(True)
    for axis in [ax_findings, ax_metrics]:
        axis.title.set_color(TEXT_DARK)

    fig.text(
        0.06,
        0.955,
        "PITCHER FATIGUE PROFILE",
        ha="left",
        fontsize=7.5,
        color=RED_SOX_RED,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.910,
        f"{pitcher_name} · {season}",
        ha="left",
        fontsize=20,
        color=RED_SOX_NAVY,
        fontweight="bold",
    )
    fig.add_artist(
        Line2D(
            [0.06, 0.97],
            [0.875, 0.875],
            transform=fig.transFigure,
            color=RED_SOX_RED,
            linewidth=2.5,
        )
    )
    fig.text(
        0.06,
        0.830,
        (
            "Regular season · Same-pitch-type, same-game early baseline · "
            f"{bundle.config.bootstrap_confidence:.0%} game-cluster bootstrap intervals"
        ),
        color=TEXT_MUTED,
        fontsize=9,
    )
    is_synthetic = (
        bundle.provenance.get("source") == "deterministic_synthetic_demo"
    )
    source_credit = (
        "Synthetic demonstration data · Design inspired by @TJStats"
        if is_synthetic
        else "Data: Baseball Savant / MLBAM via pybaseball · Design inspired by @TJStats"
    )
    fig.text(
        0.01,
        0.025,
        "Portfolio project by Neel Vadlamudi",
        color=TEXT_MUTED,
        fontsize=6.5,
        ha="left",
    )
    fig.text(
        0.5,
        0.025,
        "Historical decision support only. Not a live prediction or injury metric.",
        color=TEXT_MUTED,
        fontsize=6.5,
        style="italic",
        ha="center",
    )
    fig.text(
        0.99,
        0.025,
        source_credit,
        color=TEXT_MUTED,
        fontsize=6.5,
        ha="right",
    )

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, facecolor=WHITE)
    return fig


def summary_sheet_png_bytes(
    bundle: AnalysisBundle,
    pitcher_name: str,
    season: int,
    *,
    dpi: int = 150,
) -> bytes:
    figure = generate_summary_sheet(bundle, pitcher_name, season, dpi=dpi)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=WHITE)
    plt.close(figure)
    buffer.seek(0)
    return buffer.read()

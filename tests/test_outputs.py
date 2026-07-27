from io import BytesIO

import matplotlib
import pytest

matplotlib.use("Agg")

from PIL import Image

from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.outputs import summary_sheet_png_bytes
from pitcher_fatigue.pipeline import analyze_pitcher_frame
from pitcher_fatigue.presentation import get_display_name
from pitcher_fatigue.sample_data import make_synthetic_pitcher


def test_summary_sheet_renders_to_png():
    bundle = analyze_pitcher_frame(
        make_synthetic_pitcher(n_games=10),
        config=AnalysisConfig(min_starts=8, bootstrap_iterations=100),
        train_models=False,
    )
    image = summary_sheet_png_bytes(bundle, "Synthetic Pitcher", 2024)
    assert image.startswith(b"\x89PNG")
    assert len(image) > 20_000


def test_summary_sheet_has_web_and_print_resolutions():
    bundle = analyze_pitcher_frame(
        make_synthetic_pitcher(n_games=10),
        config=AnalysisConfig(min_starts=8, bootstrap_iterations=100),
        train_models=False,
    )
    web = Image.open(
        BytesIO(summary_sheet_png_bytes(bundle, "Synthetic Pitcher", 2024))
    )
    printed = Image.open(
        BytesIO(
            summary_sheet_png_bytes(
                bundle,
                "Synthetic Pitcher",
                2024,
                dpi=300,
            )
        )
    )
    assert web.size == (2100, 1200)
    assert printed.size == (4200, 2400)


def test_display_names_never_title_case_unknown_fields():
    assert get_display_name("spin_delta_rpm") == "Spin Rate Delta (rpm)"
    assert get_display_name("pfx_x") == "Horizontal Break (ft)"
    with pytest.warns(UserWarning):
        assert get_display_name("new_raw_feature") == "new_raw_feature"

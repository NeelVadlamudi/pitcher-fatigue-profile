from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.pipeline import analyze_pitcher_frame
from pitcher_fatigue.sample_data import make_synthetic_pitcher


def test_end_to_end_descriptive_pipeline_without_models():
    config = AnalysisConfig(
        min_starts=8,
        bootstrap_iterations=100,
    )
    bundle = analyze_pitcher_frame(
        make_synthetic_pitcher(n_games=12),
        provenance={"source": "test_synthetic"},
        config=config,
        train_models=False,
    )
    assert bundle.quality_report.is_usable
    assert bundle.coverage["games"] == 12
    assert bundle.coverage["velocity_coverage"] == 1.0
    assert "spin_rate_mnar_shift" in bundle.coverage
    assert not bundle.velocity_curve.empty
    assert bundle.threshold.status in {"established", "not_reached"}
    assert bundle.arm_stamina_index.status == "experimental"

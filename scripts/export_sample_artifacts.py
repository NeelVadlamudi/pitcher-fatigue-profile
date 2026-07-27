"""Regenerate the committed synthetic demonstration outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pitcher_fatigue.outputs import generate_summary_sheet
from pitcher_fatigue.pipeline import analyze_pitcher_frame


def main() -> None:
    raw = pd.read_csv("data/sample/sample_pitcher.csv", low_memory=False)
    bundle = analyze_pitcher_frame(
        raw,
        provenance={
            "pitcher_name": "Synthetic Demonstration Pitcher",
            "season": 2024,
            "source": "deterministic_synthetic_demo",
        },
        model_backend="sklearn",
    )
    image_path = Path("outputs/figures/synthetic_demo_summary.png")
    json_path = Path("outputs/validation/synthetic_demo_validation.json")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    generate_summary_sheet(
        bundle,
        "Synthetic Demonstration Pitcher",
        2024,
        save_path=image_path,
    )
    validation = {
        "provenance": bundle.provenance,
        "quality": bundle.quality_report.to_dict(),
        "coverage": bundle.coverage,
        "threshold": bundle.threshold.to_dict(),
        "arm_stamina_index": bundle.arm_stamina_index.to_dict(),
        "model": {
            "backend": bundle.pre_pitch_model.backend,
            "metrics": bundle.pre_pitch_model.metrics,
        },
        "importance_method": bundle.importance_method,
        "synthetic_warning": "Demonstration only; not an MLB finding.",
    }
    json_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(image_path)
    print(json_path)


if __name__ == "__main__":
    main()


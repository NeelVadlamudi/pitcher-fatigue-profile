"""Run a reproducible local validation for one real pitcher-season."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pitcher_fatigue.data_pull import load_or_pull
from pitcher_fatigue.outputs import generate_summary_sheet
from pitcher_fatigue.pipeline import analyze_pitcher_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_name")
    parser.add_argument("last_name")
    parser.add_argument("season", type=int)
    parser.add_argument("--output-dir", default="outputs/validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw, provenance = load_or_pull(
        args.first_name,
        args.last_name,
        args.season,
    )
    bundle = analyze_pitcher_frame(raw, provenance=provenance)
    slug = re.sub(
        r"[^a-z0-9]+",
        "_",
        f"{args.first_name}_{args.last_name}_{args.season}".lower(),
    ).strip("_")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{slug}_summary.png"
    json_path = output_dir / f"{slug}_validation.json"

    generate_summary_sheet(
        bundle,
        f"{args.first_name} {args.last_name}",
        args.season,
        save_path=image_path,
    )
    validation = {
        "provenance": bundle.provenance,
        "quality": bundle.quality_report.to_dict(),
        "coverage": bundle.coverage,
        "threshold": bundle.threshold.to_dict(),
        "arm_stamina_index": bundle.arm_stamina_index.to_dict(),
        "model": (
            {
                "backend": bundle.pre_pitch_model.backend,
                "metrics": bundle.pre_pitch_model.metrics,
            }
            if bundle.pre_pitch_model
            else {"error": bundle.model_error}
        ),
        "importance_method": bundle.importance_method,
    }
    json_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json_path)
    print(image_path)


if __name__ == "__main__":
    main()


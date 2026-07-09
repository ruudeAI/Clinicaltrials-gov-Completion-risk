"""
run_v2_pipeline.py - Orchestrate the Version 2 pipeline.

Runs the V2 success and duration pipeline steps in order. This script does not
modify Version 1 files; it only invokes the standalone V2 modules.

Examples
--------
    venv\\Scripts\\python.exe scripts\\run_v2_pipeline.py --max-trials 3000
    venv\\Scripts\\python.exe scripts\\run_v2_pipeline.py --skip-fetch
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(step_number: int, name: str, command: List[str]) -> None:
    """Run one pipeline step and stop immediately if it fails."""
    print("\n" + "=" * 72)
    print(f"Step {step_number}: {name}")
    print("=" * 72)
    print("Command:", " ".join(command))

    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"\nV2 pipeline stopped at step {step_number} ({name}). "
            f"Exit code: {result.returncode}"
        )


def build_steps(args: argparse.Namespace) -> List[tuple[str, List[str]]]:
    """Build the ordered V2 pipeline steps."""
    python = sys.executable
    steps: List[tuple[str, List[str]]] = []

    if not args.skip_fetch:
        steps.append(
            (
                "Fetch V2 ClinicalTrials.gov data",
                [
                    python,
                    "src/v2_clinicaltrials_api.py",
                    "--max-trials",
                    str(args.max_trials),
                ],
            )
        )

    steps.extend(
        [
            (
                "Preprocess V2 duration and endpoint features",
                [python, "src/v2_preprocess.py"],
            ),
            (
                "Create conservative phase-progression success labels",
                [python, "src/success_labels.py"],
            ),
            (
                "Enrich V2 trials with ChEMBL molecule/modality metadata",
                [python, "src/v2_enrich_chembl.py"],
            ),
            (
                "Build final V2 modeling dataset with sponsor history",
                [python, "src/v2_build_modeling_dataset.py"],
            ),
            (
                "Train V2 duration regression baseline",
                [python, "src/train_duration_model.py"],
            ),
            (
                "Train V2 success classification baseline",
                [python, "src/train_success_model.py"],
            ),
        ]
    )
    return steps


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the full V2 clinical trials pipeline.")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=300,
        help="Maximum records to fetch before drug filtering. Default: 300.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip ClinicalTrials.gov download and reuse existing raw V2 data.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the V2 pipeline."""
    args = parse_args()
    steps = build_steps(args)

    print("V2 pipeline runner")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python: {sys.executable}")
    print(f"Skip fetch: {args.skip_fetch}")
    if not args.skip_fetch:
        print(f"Max trials: {args.max_trials}")

    for index, (name, command) in enumerate(steps, start=1):
        run_step(index, name, command)

    print("\n" + "=" * 72)
    print("V2 pipeline complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()


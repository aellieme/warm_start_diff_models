"""Create one downloadable archive with logs, plots, registry, and tables."""

import argparse
import zipfile
from pathlib import Path

try:
    from .experiment_tracking import output_root
except ImportError:  # Allow direct execution: python src/experiment_tools/package_experiment_results.py
    from experiment_tracking import output_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("exp_results.zip"))
    args = parser.parse_args()
    results_root = output_root()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "bt", "checkpoints", "datasets", "experiment_tables", "graphics",
            "other_files", "service_files",
        ):
            root = results_root / name
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        archive.write(path, Path("exp_results") / path.relative_to(results_root))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

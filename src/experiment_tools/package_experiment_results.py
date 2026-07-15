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
    parser.add_argument("--output", type=Path, default=Path("experiment_results.zip"))
    args = parser.parse_args()
    results_root = output_root().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("logs", "reports", "checkpoints"):
            root = results_root / name
            if root.exists():
                for path in root.rglob("*"):
                    if path.is_file():
                        archive.write(path, Path("experiment_results") / path.relative_to(results_root))
        registry = results_root / "results_registry.csv"
        if registry.exists():
            archive.write(registry, Path("experiment_results") / registry.name)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

"""
Dataset acquisition script for Fake and Real News Dataset from Kaggle.
Downloads the dataset using kagglehub and stages files in ./data/
"""

import sys
import shutil
from pathlib import Path


def download_dataset(target_dir: str = "./data") -> Path:
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    fake_csv = target_path / "Fake.csv"
    true_csv = target_path / "True.csv"

    if fake_csv.exists() and true_csv.exists():
        print(f"Dataset already present in {target_path}:")
        print(f"  - {fake_csv.name} ({fake_csv.stat().st_size / (1024 * 1024):.2f} MB)")
        print(f"  - {true_csv.name} ({true_csv.stat().st_size / (1024 * 1024):.2f} MB)")
        return target_path

    print("Initiating dataset download via kagglehub...")
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub is not installed in the active environment.", file=sys.stderr)
        print("Please run: pip install kagglehub", file=sys.stderr)
        sys.exit(1)

    try:
        download_path = Path(
            kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
        )
        print(f"kagglehub downloaded files to: {download_path}")

        # Traverse and copy files to target_path
        found_files = []
        for file_path in download_path.rglob("*.csv"):
            destination = target_path / file_path.name
            shutil.copy2(file_path, destination)
            found_files.append(destination)
            print(f"Copied {file_path.name} -> {destination}")

        if not (fake_csv.exists() and true_csv.exists()):
            raise FileNotFoundError(
                f"Expected Fake.csv and True.csv in downloaded content. Found: {[f.name for f in found_files]}"
            )

        print(f"\nDataset successfully placed at: {target_path}")
        print(f"  - Fake.csv: {fake_csv.stat().st_size / (1024 * 1024):.2f} MB")
        print(f"  - True.csv: {true_csv.stat().st_size / (1024 * 1024):.2f} MB")
        return target_path

    except Exception as e:
        print(f"Dataset acquisition failed: {e}", file=sys.stderr)
        print(
            "Note: Ensure your network connection is active and Kaggle API is accessible.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()

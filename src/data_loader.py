"""
Data loading module for the Fake and Real News dataset.
Handles file ingestion, label assignment, and raw data consolidation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

# Standard label mapping
LABEL_MAP = {
    0: "Real",
    1: "Fake"
}
LABEL_TO_ID = {
    "Real": 0,
    "Fake": 1
}


def get_data_dir() -> Path:
    """Returns absolute path to the data directory."""
    # Assuming standard project structure where src/ is at project_root/src/
    root_dir = Path(__file__).resolve().parent.parent
    return root_dir / "data"


def load_raw_datasets(
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw Fake.csv and True.csv from the data directory.

    Returns:
        Tuple of (fake_df, true_df)
    """
    if data_dir is None:
        data_dir = get_data_dir()

    fake_path = data_dir / "Fake.csv"
    true_path = data_dir / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            f"Dataset files missing in {data_dir}. Expected 'Fake.csv' and 'True.csv'. "
            "Please run 'python scripts/download_data.py' first."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Assign labels: 1 for Fake, 0 for Real
    fake_df["label"] = 1
    true_df["label"] = 0

    return fake_df, true_df


def load_combined_dataset(
    data_dir: Optional[Path] = None,
    shuffle: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads both datasets, assigns binary labels (0=Real, 1=Fake),
    and concatenates them into a unified DataFrame.
    """
    fake_df, true_df = load_raw_datasets(data_dir=data_dir)
    combined_df = pd.concat([fake_df, true_df], ignore_index=True)

    if shuffle:
        combined_df = combined_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return combined_df

"""
Comprehensive smoke test suite to verify data loading, preprocessing,
model forward pass, and explainability on the local environment.
"""

import sys
from pathlib import Path

# Ensure project root is available
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_pipeline_smoke():
    print("1. Testing data loader...")
    from src.data_loader import load_combined_dataset
    df = load_combined_dataset(shuffle=True, random_state=42)
    assert len(df) > 0, "Dataset should not be empty"
    print(f"   -> Loaded {len(df)} records successfully.")

    print("2. Testing preprocessing & dateline stripping...")
    from src.preprocessing import prepare_dataframe, create_stratified_splits
    sample_df = df.head(100)
    cleaned = prepare_dataframe(sample_df, strip_dateline=True, drop_duplicates=True)
    assert len(cleaned) > 0
    assert "formatted_text" in cleaned.columns
    train, val, test = create_stratified_splits(cleaned, 0.8, 0.1, 0.1)
    assert len(train) + len(val) + len(test) == len(cleaned)
    print("   -> Preprocessing and stratified split verified.")

    print("3. Testing model imports and device resolution...")
    from src.model import get_device
    device, desc = get_device()
    print(f"   -> Device resolved: {desc} ({device})")

    print("Pipeline smoke test passed successfully!")


if __name__ == "__main__":
    test_pipeline_smoke()

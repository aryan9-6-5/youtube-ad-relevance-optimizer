from pathlib import Path
import subprocess
import glob

def add_dvc_files():
    """Add all Parquet files in data/processed/ to DVC."""
    processed_dir = Path(__file__).parent / 'processed'
    parquet_files = glob.glob(str(processed_dir / '*.parquet'))
    if parquet_files:
        subprocess.run(['dvc', 'add'] + parquet_files, check=True)
        print(f"✅ Added {len(parquet_files)} Parquet files to DVC")
        for f in parquet_files:
            dvc_file = Path(f).with_suffix('.dvc')
            if dvc_file.exists():
                print(f"   - {dvc_file.name}")
    else:
        print("❌ No Parquet files found in data/processed/")

if __name__ == "__main__":
    add_dvc_files()
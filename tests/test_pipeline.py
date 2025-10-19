from pathlib import Path
import sys
import pandas as pd
import duckdb
import subprocess
import glob
import yaml
import os

# Dynamic paths
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data'
processed_dir = data_dir / 'processed'
external_dir = data_dir / 'external'
db_path = data_dir / 'db' / 'youtube_ads.db'
reports_dir = project_root / 'reports'
config_path = project_root / 'config' / 'api_keys.yaml'

# Add data/ to sys.path (change to 'src/data' if scripts moved)
sys.path.append(str(data_dir))

def run_fetch():
    """Test fetch_youtube_videos.py."""
    print("\n📥 Testing fetch_youtube_videos...")
    try:
        from fetch_youtube_videos import fetch_videos_by_category
        fetch_videos_by_category('all', 500)  # Or your main function
        videos_df = pd.read_parquet(external_dir / 'real_videos.parquet')
        print(f"✅ Fetched {len(videos_df)} videos")
        print(videos_df[['title', 'views', 'likes']].head().to_string())
        return True
    except ImportError:
        print("❌ Import error: Ensure fetch_youtube_videos.py has fetch_videos_by_category()")
        return False
    except Exception as e:
        print(f"❌ Fetch failed: {e}")
        return False

def run_simulate():
    """Test simulate_realistic_behavior.py."""
    print("\n🧪 Testing simulate_realistic_behavior...")
    try:
        from simulate_realistic_behavior import main as simulate_main
        simulate_main()
        tables = ['users', 'ads', 'watches', 'impressions', 'engagements']
        for table in tables:
            df = pd.read_parquet(processed_dir / f'{table}.parquet')
            print(f"✅ {table}.parquet: {df.shape[0]:,} rows, {df.shape[1]} cols")
        return True
    except ImportError:
        print("❌ Import error: Ensure simulate_realistic_behavior.py has main()")
        return False
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

def run_validate():
    """Test validate_realistic_data.py."""
    print("\n✅ Testing validate_realistic_data...")
    try:
        from validate_realistic_data import main as validate_main
        validate_main()
        report_path = reports_dir / 'validation_report.txt'
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = f.read()
            print("📊 Validation Report (snippet):")
            print('\n'.join(report.splitlines()[:10]))
        else:
            print("❌ Report not found: Check validate_realistic_data.py output path")
        return True
    except ImportError:
        print("❌ Import error: Ensure validate_realistic_data.py has main()")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def run_ingest():
    """Test ingest_to_duckdb.py."""
    print("\n💾 Testing ingest_to_duckdb...")
    try:
        from ingest_to_duckdb import ingest_to_duckdb
        ingest_to_duckdb()
        con = duckdb.connect(str(db_path))
        result = con.execute("SELECT category, AVG(completion_pct) as avg_completion FROM raw.watches GROUP BY category ORDER BY avg_completion DESC LIMIT 5").fetchdf()
        print("✅ DuckDB Query: Average completion by category")
        print(result)
        con.close()
        return True
    except ImportError:
        print("❌ Import error: Ensure ingest_to_duckdb.py has ingest_to_duckdb()")
        return False
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False

def run_dvc():
    """Test DVC tracking."""
    print("\n📂 Testing DVC tracking...")
    try:
        from add_dvc_files import add_dvc_files
        add_dvc_files()
        dvc_files = list(processed_dir.glob('*.parquet.dvc'))
        print(f"✅ Tracked {len(dvc_files)} Parquet files with DVC")
        return True
    except ImportError:
        print("❌ Import error: Ensure add_dvc_files.py has add_dvc_files()")
        return False
    except Exception as e:
        print(f"❌ DVC tracking failed: {e}")
        return False

def test_pipeline():
    """Run full pipeline test."""
    print("🚀 Starting pipeline test (Oct 19, 2025, 12:13 PM IST)")
    reports_dir.mkdir(exist_ok=True)  # Ensure reports dir exists

    steps = [
        ('Fetch', run_fetch),
        ('Simulate', run_simulate),
        ('Validate', run_validate),
        ('Ingest', run_ingest),
        ('DVC', run_dvc)
    ]

    success_count = 0
    for step_name, step_func in steps:
        if step_func():
            success_count += 1

    print(f"\n🎉 Pipeline test complete! {success_count}/{len(steps)} steps passed.")
    if success_count == len(steps):
        print("✅ All steps successful - ready for Airflow automation!")

if __name__ == "__main__":
    test_pipeline()
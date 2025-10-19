from pathlib import Path

reports_dir = Path(r"D:\projects\youtube-ad-relevance-optimizer\reports")
reports_dir.mkdir(parents=True, exist_ok=True)

test_file = reports_dir / "test_write.txt"

try:
    with open(test_file, 'w') as f:
        f.write("Test write successful.")
    print("✅ Manual write successful!")
except Exception as e:
    print(f"❌ Manual write failed: {e}")

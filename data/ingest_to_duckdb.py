from pathlib import Path
import duckdb

DATA_DIR = Path(__file__).parent / 'processed'
DB_PATH = Path(__file__).parent / 'db' / 'youtube_ads.db'

def ingest_to_duckdb():
    try:
        con = duckdb.connect(str(DB_PATH))
        con.execute("CREATE SCHEMA IF NOT EXISTS raw")
        tables = {
            'users': """
                user_id INTEGER, age INTEGER, location VARCHAR, country VARCHAR,
                signup_date DATE, interests VARCHAR[], viewing_intensity FLOAT,
                ad_tolerance FLOAT, device_preference VARCHAR
            """,
            'ads': """
                ad_id INTEGER, advertiser VARCHAR, category VARCHAR, ad_type VARCHAR,
                daily_budget FLOAT, cpm FLOAT, target_age_min INTEGER, target_age_max INTEGER,
                created_date DATE
            """,
            'watches': """
                watch_id INTEGER, user_id INTEGER, video_id VARCHAR, category VARCHAR,
                timestamp TIMESTAMP, duration_sec FLOAT, completion_pct FLOAT, device VARCHAR,
                video_quality_score FLOAT, real_views BIGINT, real_engagement FLOAT
            """,
            'impressions': """
                impression_id INTEGER, watch_id INTEGER, user_id INTEGER, ad_id INTEGER,
                video_id VARCHAR, timestamp TIMESTAMP, ad_position VARCHAR, ad_type VARCHAR,
                category_match BOOLEAN, device VARCHAR
            """,
            'engagements': """
                engagement_id INTEGER, impression_id INTEGER, user_id INTEGER, ad_id INTEGER,
                timestamp TIMESTAMP, clicked INTEGER, skipped INTEGER, dwell_time FLOAT,
                converted INTEGER, effective_ctr FLOAT
            """
        }
        for table in tables:
            con.execute(f"DROP TABLE IF EXISTS raw.{table}")
            con.execute(f"CREATE TABLE raw.{table} ({tables[table]})")
            con.execute(f"INSERT INTO raw.{table} SELECT * FROM read_parquet('{DATA_DIR / f'{table}.parquet'}')")
            count = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
            print(f"✅ Ingested {count:,} rows to raw.{table}")
        result = con.execute("SELECT category, AVG(completion_pct) as avg_completion FROM raw.watches GROUP BY category ORDER BY avg_completion DESC").fetchdf()
        print("\nSample Query: Average completion by category")
        print(result)
        con.close()
    except PermissionError:
        print(f"❌ Permission denied writing to {DB_PATH}. Run as admin or check folder permissions.")
        raise
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        raise

if __name__ == "__main__":
    ingest_to_duckdb()
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'data'))

from fetch_youtube_videos import fetch_videos_by_category
from simulate_realistic_behavior import main as simulate_main
from validate_realistic_data import main as validate_main
from ingest_to_duckdb import ingest_to_duckdb

default_args = {
    'owner': 'youtube_ad_optimizer',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 10, 19),
}

with DAG(
    'youtube_ad_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:
    fetch_videos = PythonOperator(
        task_id='fetch_youtube_videos',
        python_callable=lambda: fetch_videos_by_category('all', 500),
    )
    simulate_data = PythonOperator(
        task_id='simulate_behavior',
        python_callable=simulate_main,
    )
    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_main,
    )
    ingest_data = PythonOperator(
        task_id='ingest_to_duckdb',
        python_callable=ingest_to_duckdb,
    )
    fetch_videos >> simulate_data >> validate_data >> ingest_data
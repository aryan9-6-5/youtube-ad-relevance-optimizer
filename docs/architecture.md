# Project Architecture

## Layers
| Layer | Description | Tools |
|-------|-------------|-------|
| Data Source | Simulated user/ad streams | Python/Kafka sim |
| Storage | Batch/event data | PostgreSQL/DuckDB/Parquet |
| Orchestration | ETL + retraining | Airflow |
| ML Lifecycle | Training/versioning | DVC + MLflow |
| Serving | Inference API | FastAPI |
| Monitoring | Metrics/drift | Prometheus + Grafana + Evidently |

## Data Flow
See README diagram.

## Scope Boundaries
- In: Simulated data only (extend to real YouTube API later)
- Out: No real-time serving yet; focus on batch weekly retrain
- Assumptions: Anonymized data; ethical ad targeting (no PII)
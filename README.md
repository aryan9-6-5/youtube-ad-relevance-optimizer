# YouTube Ad Relevance Optimizer: A Continuous Learning System

## Problem Statement
In video platforms like YouTube, ad relevance drives engagement and ROI. Traditional models stale quickly due to shifting user behaviors (trends, genres). This project builds a system that:
- Predicts P(Ad Engagement | user + ad features)
- Retrains weekly on fresh data
- Monitors drift in preferences/performance

**Goal:** Boost CTR by 20-30% via automated, adaptive targeting.

## Key Metrics
- **Primary:** Click-Through Rate (CTR) = clicks / impressions (target: >0.65%)
- **Model Performance:** ROC-AUC (target: >0.80), Precision@K for top recommendations
- **Monitoring:** Data/Model Drift (KL-Divergence <0.1 threshold), Latency (<500ms inference)
- **Business:** Churn reduction, Engagement dwell time

## System Overview
```mermaid
graph TD
    A[Simulated Data Stream] --> B[Airflow Ingestion]
    B --> C[Feature Engineering]
    C --> D[MLflow Training]
    D --> E[Model Serving API]
    E --> F[Monitoring Dashboard]

# Realistic YouTube Ad Behavioral Data Simulation

A comprehensive pipeline to generate realistic user behavioral data based on **actual YouTube video metrics**.

## 🎯 What This Does

This simulation system:
1.  Uses **real YouTube video data** (views, likes, engagement rates)
2.  Generates **realistic synthetic users** with diverse interests
3.  Models **authentic watch behavior** based on video quality scores
4.  Simulates **realistic ad impressions** with targeting logic
5.  Creates **engagement events** (clicks, skips, conversions) with real-world distributions

##  Project Structure

```
youtube-ad-relevance-optimizer/
├── data/
│   ├── external/
│   │   └── real_videos.parquet         # Real YouTube video metadata
│   └── processed/
│       ├── users.parquet                # Synthetic users
│       ├── ads.parquet                  # Synthetic ad campaigns
│       ├── watches.parquet              # Watch sessions
│       ├── impressions.parquet          # Ad impressions
│       └── engagements.parquet          # Ad engagements
├── src/
│   ├── fetch_youtube_videos.py          # Fetches real video data
│   ├── simulate_realistic_behavior.py   # Main simulation engine
│   ├── analyze_generated_data.py        # Data analysis & stats
│   └── validate_realistic_data.py       # Quality validation
└── reports/
    ├── figures/                         # Diagnostic plots
    |   ├── synthetic/                   # Synthetic data   
    |   ├── real/                        # Real data
    └── data_validation_report.txt       # Full validation report
```

##  Quick Start

### Step 1: Fetch Real YouTube Data

```bash
cd src
python fetch_youtube_videos.py
```

**Output:**
- `data/external/real_videos.parquet` (~500 real videos)
- Quota used: ~50 units (out of 10,000 daily limit)

### Step 2: Generate Realistic Behavioral Data

```bash
python simulate_realistic_behavior.py
```

**Output:**
- 2,000 users
- 500 ads
- ~15,000 watch sessions
- ~5,000 ad impressions
- ~400 ad engagements

**Time:** ~30 seconds

### Step 3: Analyze Generated Data

```bash
python analyze_generated_data.py
```

**Shows:**
- User activity patterns
- Video performance correlation with real data
- Ad effectiveness by type/category
- Temporal viewing patterns

### Step 4: Validate Data Quality

```bash
python validate_realistic_data.py
```

**Checks:**
- Data integrity (nulls, foreign keys)
- Temporal realism (peak hours)
- Behavioral patterns (completion rates)
- Industry benchmarks (CTR, skip rates)

##  Key Features

### 1. Real Data Integration
```python
# Videos have actual engagement metrics
videos['engagement_rate'] = likes / (views + 1)
videos['quality_score'] = f(engagement_rate, views)

# Watch behavior is derived from these
completion_pct = beta_dist(a=2 + quality_score, b=3)
```

### 2. Realistic User Profiles
- **Age distribution**: Matches YouTube demographics (18-65)
- **Interest diversity**: 1-3 categories per user
- **Viewing intensity**: Lognormal distribution (few power users)
- **Ad tolerance**: Beta distribution (most users have low tolerance)
- **Device preferences**: 60% mobile, 25% desktop, 10% tablet, 5% TV

### 3. Behavioral Modeling

#### Watch Patterns
-  Peak hours: 7-10 PM
-  Weekend lift: ~10
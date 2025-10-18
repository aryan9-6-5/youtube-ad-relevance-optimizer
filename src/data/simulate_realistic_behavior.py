"""
Realistic YouTube Ad Behavioral Data Simulator
Generates synthetic user interactions based on real video engagement patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import os
from pathlib import Path
from scipy import stats

fake = Faker()

# Configuration
NUM_USERS = 2000
NUM_ADS = 500
WATCH_MULTIPLIER = 3  # Avg watches per video
AD_IMPRESSION_RATE = 0.35  # 35% of watches show ads
BASE_CTR = 0.08  # 8% base click-through rate

CATEGORIES = ['tech', 'gaming', 'music', 'sports', 'food', 'travel', 'education', 'comedy']

AD_TYPES = ['pre-roll', 'mid-roll', 'banner', 'skippable', 'non-skippable', 'bumper']

# Device distributions (realistic mobile-first)
DEVICES = ['mobile', 'desktop', 'tablet', 'smart_tv']
DEVICE_WEIGHTS = [0.60, 0.25, 0.10, 0.05]


class RealisticBehaviorSimulator:
    """Generates realistic user behavior based on actual video metrics"""
    
    def __init__(self, videos_df):
        self.videos = videos_df.copy()
        self._preprocess_videos()
        
    def _preprocess_videos(self):
        """Extract behavioral signals from real video data"""
        
        # Calculate engagement rate (likes/views ratio)
        self.videos['engagement_rate'] = (
            self.videos['likes'] / (self.videos['views'] + 1)
        ).clip(0, 1)
        
        # Popularity tiers based on view quantiles
        self.videos['popularity_tier'] = pd.qcut(
            self.videos['views'], 
            q=5, 
            labels=['niche', 'emerging', 'popular', 'trending', 'viral'],
            duplicates='drop'
        )
        
        # Video quality score (composite metric)
        self.videos['quality_score'] = (
            0.7 * (self.videos['engagement_rate'] * 100) +
            0.3 * np.log1p(self.videos['views']) / np.log1p(self.videos['views'].max())
        ).clip(0, 100)
        
        # Expected watch completion (higher engagement = more complete views)
        self.videos['expected_completion'] = 0.3 + (self.videos['engagement_rate'] * 0.5)
        
        print(f"✅ Preprocessed {len(self.videos)} videos")
        print(f"   Avg engagement rate: {self.videos['engagement_rate'].mean():.2%}")
        print(f"   Quality score range: {self.videos['quality_score'].min():.1f} - {self.videos['quality_score'].max():.1f}")
    
    def generate_users(self, n=NUM_USERS):
        """Generate realistic user profiles with behavioral traits"""
        
        users = []
        for user_id in range(1, n + 1):
            # Age distribution (realistic YouTube demographics)
            age = int(np.random.choice(
                [18, 25, 35, 45, 55, 65],
                p=[0.15, 0.35, 0.25, 0.15, 0.07, 0.03]
            ) + np.random.randint(-2, 3))
            
            # Interest diversity (younger = more categories)
            num_interests = 3 if age < 30 else (2 if age < 50 else 1)
            interests = random.sample(CATEGORIES, k=num_interests)
            
            # Viewing intensity (how often they watch)
            viewing_intensity = np.random.lognormal(mean=1.5, sigma=0.8)
            
            # Ad tolerance (affects skip/engagement behavior)
            ad_tolerance = np.random.beta(a=2, b=5)  # Most users have low tolerance
            
            users.append({
                'user_id': user_id,
                'age': age,
                'location': fake.city(),
                'country': fake.country_code(),
                'signup_date': fake.date_between(start_date='-3y', end_date='-1m'),
                'interests': interests,
                'viewing_intensity': viewing_intensity,
                'ad_tolerance': ad_tolerance,
                'device_preference': np.random.choice(DEVICES, p=DEVICE_WEIGHTS)
            })
        
        df = pd.DataFrame(users)
        print(f"✅ Generated {len(df)} users")
        print(f"   Avg interests per user: {df['interests'].apply(len).mean():.1f}")
        return df
    
    def generate_ads(self, n=NUM_ADS):
        """Generate realistic ad campaigns"""
        
        ads = []
        for ad_id in range(1, n + 1):
            category = random.choice(CATEGORIES)
            ad_type = random.choice(AD_TYPES)
            
            # Budget affects targeting quality
            daily_budget = np.random.lognormal(mean=6, sigma=1.5)
            
            # Bid amounts (CPM - cost per mille)
            cpm = np.random.uniform(2, 25) if ad_type != 'bumper' else np.random.uniform(5, 15)
            
            ads.append({
                'ad_id': ad_id,
                'advertiser': fake.company(),
                'category': category,
                'ad_type': ad_type,
                'daily_budget': round(daily_budget, 2),
                'cpm': round(cpm, 2),
                'target_age_min': random.choice([18, 25, 35, 45]),
                'target_age_max': random.choice([34, 44, 54, 65]),
                'created_date': fake.date_between(start_date='-1y', end_date='-1w')
            })
        
        df = pd.DataFrame(ads)
        print(f"✅ Generated {len(df)} ads")
        return df
    
    def generate_watches(self, users_df):
        """Generate realistic watch sessions based on video engagement patterns"""
        
        watches = []
        base_date = datetime(2024, 1, 1)
        
        for _, video in self.videos.iterrows():
            # Number of watches scales with popularity
            tier_multipliers = {
                'niche': 0.5, 
                'emerging': 1.0, 
                'popular': 2.5, 
                'trending': 5.0, 
                'viral': 10.0
            }
            
            num_watches = int(
                WATCH_MULTIPLIER * 
                tier_multipliers.get(video['popularity_tier'], 1.0) *
                np.random.uniform(0.8, 1.2)  # Random variance
            )
            
            # Filter users interested in this category
            interested_users = users_df[
                users_df['interests'].apply(lambda x: video['category'] in x)
            ]
            
            if interested_users.empty:
                interested_users = users_df.sample(min(10, len(users_df)))
            
            # Weight by viewing intensity
            user_weights = interested_users['viewing_intensity'].values
            user_weights = user_weights / user_weights.sum()
            
            for _ in range(num_watches):
                user = interested_users.sample(1, weights=user_weights).iloc[0]
                
                # Timestamp with realistic patterns (peak hours: 7-10pm)
                days_offset = np.random.randint(0, 365)
                hour = int(np.random.choice(
                    range(24),
                    p=self._hourly_distribution()
                ))
                timestamp = base_date + timedelta(
                    days=days_offset,
                    hours=hour,
                    minutes=np.random.randint(0, 60)
                )
                
                # Watch completion based on video quality + user tolerance
                base_completion = video['expected_completion']
                user_factor = user['ad_tolerance'] * 0.2  # More tolerance = more completion
                
                # Beta distribution for realistic drop-off curve
                completion_pct = np.random.beta(
                    a=2 + (base_completion * 5),
                    b=3 - (base_completion * 1.5)
                )
                completion_pct = np.clip(completion_pct, 0.05, 1.0)
                
                watch_duration = completion_pct * video['duration_sec']
                
                watches.append({
                    'watch_id': len(watches) + 1,
                    'user_id': user['user_id'],
                    'video_id': video['video_id'],
                    'category': video['category'],
                    'timestamp': timestamp,
                    'duration_sec': watch_duration,
                    'completion_pct': completion_pct,
                    'device': user['device_preference'] if random.random() < 0.8 
                              else random.choice(DEVICES),
                    'video_quality_score': video['quality_score'],
                    'real_views': video['views'],  # For features
                    'real_engagement': video['engagement_rate']
                })
        
        df = pd.DataFrame(watches)
        print(f"✅ Generated {len(df)} watch sessions")
        print(f"   Avg completion: {df['completion_pct'].mean():.1%}")
        print(f"   Avg watch time: {df['duration_sec'].mean():.1f}s")
        return df
    
    def generate_ad_impressions(self, watches_df, ads_df):
        """Generate ad impressions based on watch behavior"""
        
        impressions = []
        
        for _, watch in watches_df.iterrows():
            # Ad probability increases with longer watch times
            ad_prob = AD_IMPRESSION_RATE
            if watch['completion_pct'] > 0.7:
                ad_prob *= 1.3  # More ads on complete views
            elif watch['completion_pct'] < 0.2:
                ad_prob *= 0.5  # Fewer ads on quick exits
            
            if random.random() > ad_prob:
                continue
            
            # Number of ads (depends on video length and type)
            num_ads = 1
            if watch['duration_sec'] > 180:  # 3+ min videos
                num_ads = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            
            # Select ads (prefer category match)
            category_ads = ads_df[ads_df['category'] == watch['category']]
            
            for i in range(num_ads):
                # 70% category match, 30% random
                if len(category_ads) > 0 and random.random() < 0.7:
                    ad = category_ads.sample(1).iloc[0]
                else:
                    ad = ads_df.sample(1).iloc[0]
                
                # Ad timing (pre-roll, mid-roll, etc.)
                if i == 0:
                    ad_position = 'pre-roll'
                    offset_sec = 0
                elif i == num_ads - 1 and watch['completion_pct'] > 0.9:
                    ad_position = 'post-roll'
                    offset_sec = watch['duration_sec']
                else:
                    ad_position = 'mid-roll'
                    offset_sec = random.uniform(30, watch['duration_sec'] * 0.7)
                
                impressions.append({
                    'impression_id': len(impressions) + 1,
                    'watch_id': watch['watch_id'],
                    'user_id': watch['user_id'],
                    'ad_id': ad['ad_id'],
                    'video_id': watch['video_id'],
                    'timestamp': watch['timestamp'] + timedelta(seconds=offset_sec),
                    'ad_position': ad_position,
                    'ad_type': ad['ad_type'],
                    'category_match': (ad['category'] == watch['category']),
                    'device': watch['device']
                })
        
        df = pd.DataFrame(impressions)
        print(f"✅ Generated {len(df)} ad impressions")
        print(f"   Category match rate: {df['category_match'].mean():.1%}")
        return df
    
    def generate_engagements(self, impressions_df, users_df):
        """Generate realistic ad engagement events"""
        
        engagements = []
        
        for _, imp in impressions_df.iterrows():
            user = users_df[users_df['user_id'] == imp['user_id']].iloc[0]
            
            # Base CTR varies by ad type
            ad_type_ctr = {
                'pre-roll': 0.06,
                'mid-roll': 0.08,
                'banner': 0.12,
                'skippable': 0.05,
                'non-skippable': 0.10,
                'bumper': 0.04
            }
            
            base_ctr = ad_type_ctr.get(imp['ad_type'], BASE_CTR)
            
            # Adjust CTR based on factors
            ctr = base_ctr
            
            # Category match boosts CTR by 80%
            if imp['category_match']:
                ctr *= 1.8
            
            # User ad tolerance affects engagement
            ctr *= (0.5 + user['ad_tolerance'])
            
            # Device affects CTR (mobile = lower)
            device_factors = {'mobile': 0.8, 'desktop': 1.2, 'tablet': 1.0, 'smart_tv': 0.6}
            ctr *= device_factors.get(imp['device'], 1.0)
            
            # Position matters
            if imp['ad_position'] == 'pre-roll':
                ctr *= 1.1  # Slight boost for pre-roll
            elif imp['ad_position'] == 'post-roll':
                ctr *= 0.7  # Lower for post-roll
            
            ctr = np.clip(ctr, 0.001, 0.5)  # Cap at 50% max
            
            # Determine if clicked
            clicked = 1 if random.random() < ctr else 0
            
            # Skipped (for skippable ads)
            skipped = 0
            if imp['ad_type'] in ['skippable', 'pre-roll']:
                skip_prob = 0.65 * (1 - user['ad_tolerance'])
                skipped = 1 if random.random() < skip_prob else 0
            
            # Dwell time (how long they stayed on ad/landing page)
            if clicked:
                # Lognormal distribution for dwell (most short, some long)
                dwell_time = np.random.lognormal(mean=2.5, sigma=1.0)
                dwell_time = np.clip(dwell_time, 1, 300)
            else:
                dwell_time = np.random.uniform(0, 2)
            
            # Conversion (very rare, ~2% of clicks)
            converted = 0
            if clicked and random.random() < 0.02:
                converted = 1
            
            engagements.append({
                'engagement_id': len(engagements) + 1,
                'impression_id': imp['impression_id'],
                'user_id': imp['user_id'],
                'ad_id': imp['ad_id'],
                'timestamp': imp['timestamp'],
                'clicked': clicked,
                'skipped': skipped,
                'dwell_time': round(dwell_time, 2),
                'converted': converted,
                'effective_ctr': round(ctr, 4)
            })
        
        df = pd.DataFrame(engagements)
        print(f"✅ Generated {len(df)} engagement events")
        print(f"   Overall CTR: {df['clicked'].mean():.2%}")
        print(f"   Skip rate: {df['skipped'].mean():.2%}")
        print(f"   Conversion rate: {df[df['clicked']==1]['converted'].mean():.2%}")
        return df
    
    @staticmethod
    def _hourly_distribution():
        """Realistic hourly viewing patterns (peak 7-10pm)"""
        hours = np.array([
            0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 12am-6am (low)
            0.03, 0.04, 0.05, 0.04, 0.04, 0.05,  # 6am-12pm (morning)
            0.05, 0.05, 0.04, 0.05, 0.06, 0.07,  # 12pm-6pm (afternoon)
            0.08, 0.10, 0.09, 0.08, 0.05, 0.03   # 6pm-12am (PEAK)
        ])
        return hours / hours.sum()


def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("🎬 REALISTIC YOUTUBE AD BEHAVIOR SIMULATOR")
    print("="*60)
    
    # Load real video data
    video_path = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "real_videos.parquet"
    if not video_path.exists():
        print("❌ Real video data not found. Run fetch_youtube_videos.py first!")
        return
    
    videos = pd.read_parquet(video_path)
    print(f"\n📥 Loaded {len(videos)} real YouTube videos")
    
    # Initialize simulator
    sim = RealisticBehaviorSimulator(videos)
    
    # Generate all datasets
    print("\n🔄 Generating synthetic behavioral data...\n")
    
    users = sim.generate_users()
    ads = sim.generate_ads()
    watches = sim.generate_watches(users)
    impressions = sim.generate_ad_impressions(watches, ads)
    engagements = sim.generate_engagements(impressions, users)
    
    # Save outputs
    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving datasets to {output_dir}/...")
    
    users.to_parquet(output_dir / "users.parquet", index=False)
    ads.to_parquet(output_dir / "ads.parquet", index=False)
    watches.to_parquet(output_dir / "watches.parquet", index=False)
    impressions.to_parquet(output_dir / "impressions.parquet", index=False)
    engagements.to_parquet(output_dir / "engagements.parquet", index=False)
    
    # Also save CSVs for easy inspection
    watches.to_csv(output_dir / "watches.csv", index=False)
    impressions.to_csv(output_dir / "impressions.csv", index=False)
    engagements.to_csv(output_dir / "engagements.csv", index=False)
    
    # Summary statistics
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Users:        {len(users):,}")
    print(f"Ads:          {len(ads):,}")
    print(f"Videos:       {len(videos):,}")
    print(f"Watches:      {len(watches):,}")
    print(f"Impressions:  {len(impressions):,}")
    print(f"Engagements:  {len(engagements):,}")
    print(f"\nAvg watches/video:      {len(watches)/len(videos):.1f}")
    print(f"Avg impressions/watch:  {len(impressions)/len(watches):.2f}")
    print(f"Overall CTR:            {engagements['clicked'].mean():.2%}")
    print(f"Category match impact:  +{((impressions[impressions['category_match']].merge(engagements, on='impression_id')['clicked'].mean() / engagements['clicked'].mean() - 1) * 100):.0f}%")
    
    print("\n✅ All datasets generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
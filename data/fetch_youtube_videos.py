import os
import yaml
from googleapiclient.discovery import build
import pandas as pd
from pathlib import Path
import yaml               # Also ensure you are importing yaml

# Load API key
config_path = Path(__file__).parent.parent / 'config' / 'api_keys.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)
API_KEY = config['youtube']['api_key']
YOUTUBE = build('youtube', 'v3', developerKey=API_KEY)

CATEGORIES = ['tech', 'gaming', 'music', 'sports', 'food', 'travel', 'education', 'comedy']  # Our focus
NUM_VIDEOS = 500  # ~5 pages of 100 results (500 units quota)
MAX_RESULTS = 50  # Per page to chunk calls

def fetch_videos_by_category(category, max_videos):
    videos = []
    next_page_token = None
    fetched = 0

    while fetched < max_videos:
        # Step 1: Search for videos (only 'snippet')
        search_response = YOUTUBE.search().list(
            q=f"{category} review 2025",
            part='snippet',
            type='video',
            maxResults=min(MAX_RESULTS, max_videos - fetched),
            pageToken=next_page_token,
            order='viewCount'
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response['items']]
        next_page_token = search_response.get('nextPageToken')
        if not video_ids:
            break

        # Step 2: Fetch statistics using videos().list()
        video_response = YOUTUBE.videos().list(
            part='statistics,snippet',
            id=','.join(video_ids)
        ).execute()

        for item in video_response['items']:
            snippet = item['snippet']
            stats = item.get('statistics', {})
            videos.append({
                'video_id': item['id'],
                'title': snippet['title'],
                'description': snippet.get('description', '')[:200],
                'category': category,
                'publish_date': snippet['publishedAt'],
                'channel_id': snippet['channelId'],
                'channel_title': snippet['channelTitle'],
                'views': int(stats.get('viewCount', 0)),
                'likes': int(stats.get('likeCount', 0)),
                'duration_sec': 300  # Placeholder
            })
            fetched += 1
            if fetched >= max_videos:
                break

        if not next_page_token:
            break

    return pd.DataFrame(videos)

if __name__ == "__main__":
    print("Fetching real YouTube videos...")
    all_videos = pd.DataFrame()
    videos_per_cat = NUM_VIDEOS // len(CATEGORIES)

    for cat in CATEGORIES:
        cat_videos = fetch_videos_by_category(cat, videos_per_cat)
        all_videos = pd.concat([all_videos, cat_videos], ignore_index=True)
        print(f"Fetched {len(cat_videos)} {cat} videos")

    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'external' / 'real_videos.parquet'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'real_videos.parquet'
    all_videos.to_parquet(output_path, index=False)

    print(f"Total: {len(all_videos)} videos saved to {output_path}")
    print(f"Quota tip: ~{len(CATEGORIES)} calls = {NUM_VIDEOS / 10} units used.")
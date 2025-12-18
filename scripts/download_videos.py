"""
YouTube Video Downloader
Download Clash Royale gameplay videos for imitation learning
"""

import argparse
import subprocess
from pathlib import Path
from typing import List


def download_videos(
    query: str,
    output_dir: str = "data/videos",
    max_count: int = 50,
    max_duration: int = 600  # 10 minutes
):
    """
    Download videos from YouTube
    
    Args:
        query: Search query
        output_dir: Output directory
        max_count: Maximum number of videos to download
        max_duration: Maximum video duration in seconds
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading videos for query: '{query}'")
    print(f"Output directory: {output_path}")
    print(f"Max count: {max_count}")
    
    # yt-dlp command
    # Search for videos and download
    cmd = [
        "yt-dlp",
        f"ytsearch{max_count}:{query}",
        "--output", str(output_path / "%(title)s.%(ext)s"),
        "--format", "best[height<=720]",  # 720p max for faster download
        "--match-filter", f"duration < {max_duration}",
        "--no-playlist",
        "--write-info-json",  # Save metadata
        "--write-thumbnail",  # Save thumbnail
    ]
    
    try:
        print("\nStarting download...")
        subprocess.run(cmd, check=True)
        print("\n✓ Download complete!")
        
        # Count downloaded videos
        videos = list(output_path.glob("*.mp4")) + list(output_path.glob("*.webm"))
        print(f"Downloaded {len(videos)} videos")
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
    except FileNotFoundError:
        print("\n✗ yt-dlp not found. Please install it:")
        print("  pip install yt-dlp")


def list_videos(video_dir: str = "data/videos") -> List[Path]:
    """
    List downloaded videos
    
    Args:
        video_dir: Video directory
    
    Returns:
        List of video paths
    """
    video_path = Path(video_dir)
    
    if not video_path.exists():
        print(f"Directory not found: {video_path}")
        return []
    
    videos = list(video_path.glob("*.mp4")) + list(video_path.glob("*.webm"))
    
    print(f"\n=== Videos in {video_path} ===")
    for i, video in enumerate(videos, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"{i:3d}. {video.name:60s} ({size_mb:.1f} MB)")
    
    print(f"\nTotal: {len(videos)} videos")
    
    return videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Clash Royale gameplay videos")
    
    parser.add_argument(
        "--query",
        type=str,
        default="clash royale arena 1 gameplay",
        help="YouTube search query"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/videos",
        help="Output directory"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Maximum number of videos to download"
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=600,
        help="Maximum video duration in seconds"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List downloaded videos"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_videos(args.output)
    else:
        download_videos(
            query=args.query,
            output_dir=args.output,
            max_count=args.count,
            max_duration=args.max_duration
        )

#!/usr/bin/env python3
# From https://github.com/jfgonsalves/parakeet-diarized (commit 6abadfd)
# Copyright (c) jfgonsalves - MIT License
# String updates for NoobScribe by meganoob1337
import os
import sys
import argparse
import requests
import json
from pprint import pprint
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(description="Test the NoobScribe API (Whisper-compatible)")
    parser.add_argument("--file", required=True, help="Path to audio file to transcribe")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--format", default="verbose_json", choices=["json", "text", "srt", "vtt", "verbose_json"], 
                        help="Response format")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    args = parser.parse_args()
    
    # Check if file exists
    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"Error: File {args.file} does not exist.")
        sys.exit(1)
    
    # Construct API URL
    api_url = f"{args.url}/v1/audio/transcriptions"
    print(f"Testing API at: {api_url}")
    
    # Create test cases
    test_cases = [
        {
            "name": f"Basic transcription ({args.format} format)",
            "params": {
                "model": "whisper-1",
                "response_format": args.format,
                "timestamps": "true" if args.timestamps else "false",
                "diarize": "true" if args.diarize else "false"
            }
        },
        {
            "name": "Transcription with timestamp granularities",
            "params": {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
                "diarize": "true" if args.diarize else "false"
            }
        }
    ]
    
    # Run tests
    for test in test_cases:
        print(f"\n\nRunning test: {test['name']}")
        print("-" * 80)
        
        try:
            # Prepare the file and form data
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, f"audio/{audio_path.suffix[1:]}")}
                
                # Add form parameters
                start_time = time.time()
                response = requests.post(api_url, files=files, data=test["params"])
                elapsed = time.time() - start_time
                
                print(f"Status code: {response.status_code}")
                print(f"Time taken: {elapsed:.2f} seconds")
                
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "")
                    
                    if "json" in content_type:
                        result = response.json()
                        # Pretty print with truncated text
                        if "text" in result and len(result["text"]) > 100:
                            text_preview = result["text"][:100] + "..."
                            result_copy = result.copy()
                            result_copy["text"] = text_preview
                            pprint(result_copy)
                            print(f"\nFull text ({len(result['text'])} chars):")
                            print(result["text"])
                        else:
                            pprint(result)
                            
                        # Check if segments exist
                        if "segments" in result:
                            print(f"\nFound {len(result['segments'])} segments")
                            if result["segments"]:
                                print("First segment:")
                                pprint(result["segments"][0])
                    else:
                        # For text formats, print a preview
                        text = response.text
                        print(f"Response text preview ({len(text)} chars):")
                        print(text[:500] + ("..." if len(text) > 500 else ""))
                else:
                    print(f"Error response: {response.text}")
        
        except Exception as e:
            print(f"Error during test: {str(e)}")
    
    # Test health endpoint
    print("\n\nTesting /health endpoint")
    print("-" * 80)
    try:
        response = requests.get(f"{args.url}/health")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            pprint(response.json())
    except Exception as e:
        print(f"Error during health check: {str(e)}")
        
    # Test models endpoint
    print("\n\nTesting /v1/models endpoint")
    print("-" * 80)
    try:
        response = requests.get(f"{args.url}/v1/models")
        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            pprint(response.json())
    except Exception as e:
        print(f"Error during models check: {str(e)}")

if __name__ == "__main__":
    main()
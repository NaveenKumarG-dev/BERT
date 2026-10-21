"""
Robust downloader for BERT model weights with chunked streaming,
extended read timeouts, resume support, and progress reporting.
"""

from pathlib import Path
import sys
import time
import requests

MODEL_URL = "https://huggingface.co/bert-base-uncased/resolve/main/model.safetensors"
SNAPSHOT_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--bert-base-uncased" / "snapshots" / "86b5e0934494bd15c9632b12f734a8a67f723594"


def download_model_safetensors():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    target_file = SNAPSHOT_DIR / "model.safetensors"
    part_file = SNAPSHOT_DIR / "model.safetensors.part"

    if target_file.exists() and target_file.stat().st_size > 400_000_000:
        print(f"model.safetensors already exists ({target_file.stat().st_size / (1024*1024):.2f} MB)")
        return target_file

    downloaded = part_file.stat().st_size if part_file.exists() else 0
    print(f"Target file: {target_file}")
    print(f"Resuming from: {downloaded / (1024*1024):.2f} MB")

    headers = {}
    if downloaded > 0:
        headers["Range"] = f"bytes={downloaded}-"

    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"Connecting to Hugging Face CDN (attempt {retry_count + 1}/{max_retries})...")
            response = requests.get(
                MODEL_URL,
                headers=headers,
                stream=True,
                timeout=(15, 60),  # 15s connect, 60s read per chunk
            )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0)) + downloaded
            mode = "ab" if downloaded > 0 else "wb"

            with open(part_file, mode) as f:
                start_time = time.time()
                last_report = start_time
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunk
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_report >= 5.0:
                            pct = (downloaded / total_size * 100) if total_size else 0
                            mb = downloaded / (1024 * 1024)
                            speed = (downloaded / (now - start_time)) / (1024 * 1024)
                            print(f"Progress: {mb:.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s")
                            last_report = now

            # Completed successfully
            if part_file.exists() and part_file.stat().st_size >= 440_000_000:
                part_file.rename(target_file)
                print(f"\nDownload completed: {target_file} ({target_file.stat().st_size / (1024*1024):.2f} MB)")
                return target_file

        except Exception as e:
            print(f"Network interruption: {e}. Retrying in 3 seconds...", file=sys.stderr)
            retry_count += 1
            time.sleep(3)
            downloaded = part_file.stat().st_size if part_file.exists() else 0
            headers["Range"] = f"bytes={downloaded}-"

    print("Failed to download model weights after max retries.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    download_model_safetensors()

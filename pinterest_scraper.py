import requests
import ssl
import requests.adapters
import os
from pinscrape import Pinterest
import time
import hashlib

# -----------------------------
# TLS FIX
# -----------------------------
ssl_context = ssl.create_default_context()
ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')

class SSLAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = ssl_context
        return super().proxy_manager_for(*args, **kwargs)

s = requests.Session()
s.mount("https://", SSLAdapter())

# -----------------------------
# SETTINGS
# -----------------------------
keyword = "sikku kolam"
output_folder = "output"
chunk_size = 200      # how many images per request
target_total = 3000   # total images you want
number_of_workers = 10

os.makedirs(output_folder, exist_ok=True)

# Track already downloaded via MD5 hash
downloaded_hashes = set()

def unique_filename(url):
    """Make a deterministic unique filename from URL."""
    h = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(output_folder, f"{h}.jpg")

# -----------------------------
# SCRAPER LOOP
# -----------------------------
p = Pinterest()
downloaded_count = 0
attempt = 0

while downloaded_count < target_total:
    attempt += 1
    print(f"\n[Attempt {attempt}] Fetching next {chunk_size} images...")

    try:
        images_url = p.search(keyword, chunk_size)
    except Exception as e:
        print(f"Search failed: {e}")
        time.sleep(10)
        continue

    if not images_url:
        print("No more results returned, stopping.")
        break

    unique_urls = []
    for url in images_url:
        fname = unique_filename(url)
        if not os.path.exists(fname) and fname not in downloaded_hashes:
            unique_urls.append(url)
            downloaded_hashes.add(fname)

    if not unique_urls:
        print("All fetched URLs were duplicates.")
        break

    try:
        p.download(
            url_list=unique_urls,
            number_of_workers=number_of_workers,
            output_folder=output_folder
        )
    except Exception as e:
        print(f"Download error: {e}")
        time.sleep(10)
        continue

    downloaded_count += len(unique_urls)
    print(f"✅ Downloaded so far: {downloaded_count}/{target_total}")

    # polite delay to avoid rate-limit
    time.sleep(5)

print(f"\n🎉 Finished! Total downloaded: {downloaded_count}")

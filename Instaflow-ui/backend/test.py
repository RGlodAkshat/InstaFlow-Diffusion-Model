import os
import sys
import time
import json
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE = "http://localhost:8000"
OUT = Path(__file__).parent / "tests_out"
OUT.mkdir(exist_ok=True)

SAFE_PROMPT = "a scenic mountain landscape with a clear blue lake and pine trees, ultra-detailed, photorealistic"
W, H = 512, 512
SEED = 123

def save_image(base_url: str, image_url: str, fname_hint: str) -> Path:
    """Download image_url (can be relative like /images/abc.png) into tests_out/."""
    full = image_url if image_url.startswith("http") else urljoin(base_url, image_url)
    r = requests.get(full, timeout=120)
    r.raise_for_status()
    # Create a nice file name
    ext = ".png"
    name = f"{int(time.time())}_{fname_hint}{ext}"
    fpath = OUT / name
    with open(fpath, "wb") as f:
        f.write(r.content)
    return fpath

def pretty(d):
    print(json.dumps(d, indent=2))

def step(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def main():
    # 1) Health check
    step("1) /healthz")
    r = requests.get(f"{BASE}/healthz", timeout=30)
    r.raise_for_status()
    pretty(r.json())

    # 2) Generate (baseline)
    step("2) /generate (baseline)")
    payload = {"prompt": SAFE_PROMPT, "model": "baseline", "width": W, "height": H, "seed": SEED}
    r = requests.post(f"{BASE}/generate", json=payload, timeout=600)
    r.raise_for_status()
    out = r.json()
    pretty(out)
    try:
        f = save_image(BASE, out["image_url"], "baseline")
        print(f"Saved baseline → {f}")
    except Exception as e:
        print(f"Could not download baseline image: {e}")

    # 3) Generate (quantized)
    step("3) /generate (quantized)")
    payload = {"prompt": SAFE_PROMPT, "model": "quantized", "width": W, "height": H, "seed": SEED}
    r = requests.post(f"{BASE}/generate", json=payload, timeout=600)
    r.raise_for_status()
    out = r.json()
    pretty(out)
    try:
        f = save_image(BASE, out["image_url"], "quantized")
        print(f"Saved quantized → {f}")
    except Exception as e:
        print(f"Could not download quantized image: {e}")

    # 4) Compare
    step("4) /compare (both)")
    payload = {"prompt": SAFE_PROMPT, "width": W, "height": H, "seed": SEED}
    r = requests.post(f"{BASE}/compare", json=payload, timeout=1200)
    r.raise_for_status()
    out = r.json()
    pretty(out)
    try:
        f1 = save_image(BASE, out["baseline"]["image_url"], "compare_baseline")
        f2 = save_image(BASE, out["quantized"]["image_url"], "compare_quantized")
        print(f"Saved compare images → {f1.name}, {f2.name}")
    except Exception as e:
        print(f"Could not download compare images: {e}")

    print("\nAll tests done. Check the numbers and images in:", OUT)

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP error:", e.response.status_code, e.response.text, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

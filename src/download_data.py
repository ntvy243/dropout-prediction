"""Download the UCI dataset 697 (Realinho et al., 2022) into data/data.csv if it is missing.
Usage: python src/download_data.py [--force]"""
import io, sys, zipfile, urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "data.csv"

def main(force=False):
    if OUT.exists() and not force:
        print("data/data.csv already present"); return
    OUT.parent.mkdir(exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as r: z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.endswith("data.csv"))
    OUT.write_bytes(z.read(name)); print("saved", OUT, f"({OUT.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main("--force" in sys.argv)

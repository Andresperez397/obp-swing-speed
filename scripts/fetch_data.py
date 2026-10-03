"""Download the OpenBiomechanics hitting files used by this project.

Data: The OpenBiomechanics Project (Driveline Baseball), CC BY-NC-SA 4.0.
https://github.com/drivelineresearch/openbiomechanics
The raw files are not redistributed in this repository; run this script instead.
"""
import hashlib
import shutil
import ssl
from pathlib import Path
from urllib.request import urlopen

import certifi

OBP_COMMIT = "44b98dae05cceb016f080ab39d105c85b8639084"  # pinned so results stay reproducible
BASE = f"https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/{OBP_COMMIT}/baseball_hitting/data"
FILES = {
    "poi_metrics.csv": f"{BASE}/poi/poi_metrics.csv",
    "hittrax.csv": f"{BASE}/poi/hittrax.csv",
    "metadata.csv": f"{BASE}/metadata.csv",
    "data_dictionary.csv": f"{BASE}/data_dictionary.csv",
}
# python.org builds of Python on macOS ship without system root certificates; use certifi's.
SSL = ssl.create_default_context(cafile=certifi.where())


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        with urlopen(url, timeout=60, context=SSL) as r, open(out / name, "wb") as f:
            shutil.copyfileobj(r, f)
        digest = hashlib.sha256((out / name).read_bytes()).hexdigest()[:16]
        print(f"saved {name}  sha256:{digest}")


if __name__ == "__main__":
    main()

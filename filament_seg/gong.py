"""Download unlabeled GONG H-alpha frames for self-supervised pre-training (images only, no labels).

Source: the public NSO GONG archive (https://gong2.nso.edu/HA/hag/). 1,500 random days in 2011-2022 (seed 0),
up to two frames per day from different stations, preferring 10-20 UT; frames whose name appears in the
competition train or test set are skipped. One connection, 0.5 s between requests; re-running resumes.

The host requires that no unlabeled frame lies within +-1 h of a test image. `clean_frames` enforces this
conservatively: every frame within +-24 h of a test observation (or on the same UT day) is dropped before SSL.
"""
import random
import re
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path


def _get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def download(dst: Path, exclude: set, n_days=1500, per_day=2, log=print) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    d0 = date(2011, 1, 1)
    days = sorted(rng.sample(range((date(2022, 12, 31) - d0).days), n_days))
    got = 0
    for k, dd in enumerate(days):
        d = d0 + timedelta(days=dd)
        base = f"https://gong2.nso.edu/HA/hag/{d:%Y%m}/{d:%Y%m%d}/"
        try:
            names = sorted(set(re.findall(rf"({d:%Y%m%d}\d{{6}}[A-Z]h)\.jpg", _get(base).decode("latin1"))))
        except Exception:
            continue
        names = [n for n in names if n not in exclude and 10 <= int(n[8:10]) <= 20] or [n for n in names if n not in exclude]
        rng.shuffle(names)
        picked, stations = [], set()
        for n in names:
            if n[14] in stations:
                continue
            picked.append(n)
            stations.add(n[14])
            if len(picked) >= per_day:
                break
        for n in picked:
            f = dst / f"{n}.jpg"
            if f.exists():
                got += 1
                continue
            try:
                f.write_bytes(_get(base + n + ".jpg", 60))
                got += 1
            except Exception as e:
                log(f"fail {n} {e}")
            time.sleep(0.5)
        if k % 50 == 0:
            log(f"day {k}/{len(days)} got {got}")
    return got


def obs_time(name: str) -> datetime:
    """Observation time (UT) from a GONG / competition file name (YYYYMMDDhhmmss...)."""
    return datetime.strptime(Path(name).name[:14], "%Y%m%d%H%M%S")


def clean_frames(files, test_names, window_h=24.0):
    """Drop frames within `window_h` hours of, or on the same UT day as, any test image. Asserts none within +-1 h remain."""
    tt = [obs_time(n) for n in test_names]
    days = {t.date() for t in tt}
    keep = []
    for f in files:
        t = obs_time(f)
        dh = min(abs((t - x).total_seconds()) for x in tt) / 3600
        if dh > window_h and t.date() not in days:
            keep.append(f)
    assert all(min(abs((obs_time(f) - x).total_seconds()) for x in tt) > 3600 for f in keep)
    return keep

"""The SSL frame filter reproduces the bundled clean list (resources/), and no kept frame is within 1 h of a test frame."""
import os
from pathlib import Path

import pandas as pd
import pytest

from filament_seg import gong, io

RES = Path(__file__).resolve().parents[1] / "resources"
TEST_DIR = Path(os.environ.get("SOLAR_DATA", Path(__file__).resolve().parents[1] / "data" / "MAGFiLO_1.0_Kaggle_2026")) / "test" / "test_images"


def test_audit_table_matches_clean_list():
    audit = pd.read_csv(RES / "gong_exclusion_audit.tsv", sep="\t")
    clean = (RES / "gong_unlabeled_clean.txt").read_text().split()
    assert len(audit) == 3000 and len(clean) == 2780
    assert sorted(audit.loc[audit.excluded == 0, "gong_frame"]) == sorted(clean)
    assert (audit.loc[audit.excluded == 0, "delta_h"] > 24).all()
    assert (audit.delta_h <= 1).sum() == 19  # 19 downloaded frames lie within 1 h of a test frame; all 19 are excluded


@pytest.mark.skipif(not TEST_DIR.exists(), reason="needs the test image names (SOLAR_DATA)")
def test_clean_frames_reproduces_the_list():
    audit = pd.read_csv(RES / "gong_exclusion_audit.tsv", sep="\t")
    kept = gong.clean_frames(list(audit.gong_frame), [f.stem for f in io.list_images(TEST_DIR)])
    assert sorted(kept) == sorted((RES / "gong_unlabeled_clean.txt").read_text().split())

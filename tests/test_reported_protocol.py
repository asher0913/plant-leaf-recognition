"""The reported numbers, the README and the CLI must agree on the split protocol."""

import json
from pathlib import Path

from leafnet.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_reported_numbers_carry_their_protocol():
    protocol = json.loads((ROOT / "results" / "reported_results.json").read_text())["protocol"]
    assert protocol["splits"] == {"stratified": False, "test_size": 0.3, "random_state": "0-19"}
    assert "not reproduced" in protocol["status"]
    assert "--no-stratify" in protocol["same_protocol_with_this_code"]


def test_default_split_is_stratified_and_the_reported_one_can_be_restored():
    parser = build_parser()
    assert parser.parse_args(["run", "--data-dir", "d"]).no_stratify is False
    assert parser.parse_args(["run", "--data-dir", "d", "--no-stratify"]).no_stratify is True


def test_readme_labels_the_headline_as_historical():
    readme = (ROOT / "README.md").read_text()
    assert "unstratified" in readme and "historical numbers reported from the original campaign" in readme
    assert "98.46 ± 0.71" not in readme.split("## Quick start")[1]  # no fake reproduction output

import json

import pytest

from leafnet.cli import main
from leafnet.engine import TrainConfig
from leafnet.experiment import ExperimentConfig, run_repeated_holdout


def quiet(cfg, factory):
    return run_repeated_holdout(cfg, model_factory=factory, log=lambda _: None)


def config(leaf_folder, tmp_path, **overrides):
    values = dict(
        data_dir=str(leaf_folder),
        output_dir=str(tmp_path / "run"),
        repeats=2,
        image_size=16,
        batch_size=8,
        num_workers=0,
        pretrained=False,
        device="cpu",
        train=TrainConfig(epochs=30, lr=5e-2, eval_epochs=(10, 30)),
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def test_end_to_end_learns_separable_species(leaf_folder, tmp_path, tiny_factory):
    report = quiet(config(leaf_folder, tmp_path), tiny_factory)
    assert report["num_classes"] == 3
    assert [row["split"] for row in report["splits"]] == [0, 1]
    for row in report["splits"]:
        assert row["train_size"] + row["test_size"] == 24
        assert row["losses"][-1] < row["losses"][0]
        assert set(row["checkpoints"]) == {"10", "30"}
    assert report["summary"]["top1"]["mean"] >= 0.9
    saved = json.loads((tmp_path / "run" / "results.json").read_text())
    assert saved["summary"] == report["summary"]


def test_rerun_resumes_completed_splits(leaf_folder, tmp_path, tiny_factory):
    quiet(config(leaf_folder, tmp_path, repeats=1), tiny_factory)
    messages = []
    report = run_repeated_holdout(
        config(leaf_folder, tmp_path, repeats=2), model_factory=tiny_factory, log=messages.append
    )
    assert any("split 0: already complete" in m for m in messages)
    assert [row["split"] for row in report["splits"]] == [0, 1]


def test_changed_protocol_refuses_to_mix_results(leaf_folder, tmp_path, tiny_factory):
    quiet(config(leaf_folder, tmp_path, repeats=1), tiny_factory)
    with pytest.raises(ValueError, match="different protocol"):
        run_repeated_holdout(
            config(leaf_folder, tmp_path, repeats=1, test_size=0.5),
            model_factory=tiny_factory,
            log=lambda _: None,
        )


def test_summarize_command_prints_table(leaf_folder, tmp_path, tiny_factory, capsys):
    quiet(config(leaf_folder, tmp_path, repeats=1), tiny_factory)
    assert main(["summarize", str(tmp_path / "run" / "results.json")]) == 0
    out = capsys.readouterr().out
    assert "top1" in out and "f1_weighted" in out

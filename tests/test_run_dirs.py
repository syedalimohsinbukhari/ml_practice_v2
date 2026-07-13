from datetime import datetime

from gwml.training.train import create_run_dir, latest_run_dir


def test_create_run_dir_uses_filesystem_safe_timestamp(tmp_path):
    cfg = {"name": "resnet1d", "run_dir": str(tmp_path / "resnet1d")}
    now = datetime(2026, 7, 13, 15, 30, 12)

    run_dir = create_run_dir(cfg, now=now)

    assert run_dir == tmp_path / "resnet1d" / "20260713_153012"
    assert run_dir.is_dir()


def test_create_run_dir_adds_suffix_for_same_second_collision(tmp_path):
    cfg = {"name": "resnet1d", "run_dir": str(tmp_path / "resnet1d")}
    now = datetime(2026, 7, 13, 15, 30, 12)
    create_run_dir(cfg, now=now)

    run_dir = create_run_dir(cfg, now=now)

    assert run_dir == tmp_path / "resnet1d" / "20260713_153012_01"
    assert run_dir.is_dir()


def test_latest_run_dir_prefers_legacy_flat_run(tmp_path):
    run_dir = tmp_path / "resnet1d"
    run_dir.mkdir()
    (run_dir / "best.weights.h5").touch()
    (run_dir / "20260713_153012").mkdir()

    assert latest_run_dir({"name": "resnet1d", "run_dir": str(run_dir)}) == run_dir


def test_latest_run_dir_returns_newest_timestamped_child(tmp_path):
    run_dir = tmp_path / "resnet1d"
    older = run_dir / "20260713_153012"
    newer = run_dir / "20260713_153013"
    older.mkdir(parents=True)
    newer.mkdir()

    assert latest_run_dir({"name": "resnet1d", "run_dir": str(run_dir)}) == newer

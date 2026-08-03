from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_run_transformer.py"
SPEC = importlib.util.spec_from_file_location("run_transformer", SCRIPT_PATH)
assert SPEC and SPEC.loader
RUN_TRANSFORMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_TRANSFORMER)


class _FakeCuda:
    def __init__(self, available: bool, devices: int = 0):
        self._available = available
        self._devices = devices

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._devices


class _FakeMps:
    def __init__(self, available: bool, built: bool):
        self._available = available
        self._built = built

    def is_available(self) -> bool:
        return self._available

    def is_built(self) -> bool:
        return self._built


def test_accelerator_status_prefers_cuda_and_reports_cpu_fallback():
    cuda_torch = SimpleNamespace(
        cuda=_FakeCuda(True, devices=2),
        backends=SimpleNamespace(mps=_FakeMps(True, built=True)),
    )
    assert RUN_TRANSFORMER._accelerator_status(cuda_torch) == {
        "backend": "cuda",
        "cuda_available": True,
        "cuda_device_count": 2,
        "mps_available": True,
        "mps_built": True,
    }

    cpu_torch = SimpleNamespace(
        cuda=_FakeCuda(False),
        backends=SimpleNamespace(mps=_FakeMps(False, built=True)),
    )
    assert RUN_TRANSFORMER._accelerator_status(cpu_torch)["backend"] == "cpu"


def test_planned_experiment_count_uses_manifest_dataset_coverage(tmp_path):
    split_root = tmp_path / "splits"
    split_root.mkdir()
    pd.DataFrame({"dataset": ["coaid", "fakehealth", "fakehealth"]}).to_csv(
        split_root / "standard_split_manifest.csv",
        index=False,
    )
    pd.DataFrame({"dataset": ["fakehealth"]}).to_csv(
        split_root / "controlled_split_manifest.csv",
        index=False,
    )

    count, plan = RUN_TRANSFORMER._planned_experiment_count(
        {"splits": split_root},
        [{"experiment_name": "a"}, {"experiment_name": "b"}],
        [613, 614, 615],
    )

    assert count == 18
    assert plan[0]["datasets"] == ["coaid", "fakehealth"]
    assert plan[1]["datasets"] == ["fakehealth"]
    assert plan[2]["datasets"] == []
    assert plan[3]["datasets"] == []

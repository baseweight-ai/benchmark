"""Tests for pipeline.hardware.check_allowed_gpu enforcement."""
import pytest

from pipeline.hardware import (
    GpuMismatchError,
    _load_allowed_gpus,
    check_allowed_gpu,
    get_current_gpu_name,
)

pytestmark = pytest.mark.unit


# ── _load_allowed_gpus ────────────────────────────────────────────────────────

def test_load_allowed_gpus_reads_pricing_yaml(tmp_path, monkeypatch):
    """Reads self_hosted.allowed_gpus from pricing.yaml."""
    import pipeline.hardware as hw
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pricing.yaml").write_text(
        "apis: {}\n"
        "self_hosted:\n"
        "  allowed_gpus:\n"
        "    - 'NVIDIA GeForce RTX 3090'\n"
        "    - 'NVIDIA A10G'\n"
    )
    monkeypatch.setattr(hw, "REPO_ROOT", tmp_path)
    assert _load_allowed_gpus() == ["NVIDIA GeForce RTX 3090", "NVIDIA A10G"]


def test_load_allowed_gpus_empty_when_unset(tmp_path, monkeypatch):
    """Missing or empty allowed_gpus → enforcement disabled."""
    import pipeline.hardware as hw
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "pricing.yaml").write_text("apis: {}\nself_hosted: {}\n")
    monkeypatch.setattr(hw, "REPO_ROOT", tmp_path)
    assert _load_allowed_gpus() == []


# ── check_allowed_gpu ─────────────────────────────────────────────────────────

def test_check_skip_bypasses_validation():
    """skip=True (smoke-test / dry-run) is a hard bypass — no GPU check."""
    # Even if allowlist is empty AND no GPU exists, skip=True must not raise.
    check_allowed_gpu(skip=True)  # must not raise


def test_check_no_allowlist_skips_validation(monkeypatch):
    """Empty allowlist disables enforcement (dev iteration)."""
    monkeypatch.setattr("pipeline.hardware._load_allowed_gpus", lambda: [])
    check_allowed_gpu(skip=False)  # must not raise


def test_check_no_gpu_silently_returns(monkeypatch):
    """When no CUDA device is available, defer to downstream errors."""
    monkeypatch.setattr("pipeline.hardware._load_allowed_gpus", lambda: ["A100"])
    monkeypatch.setattr("pipeline.hardware.get_current_gpu_name", lambda: None)
    check_allowed_gpu(skip=False)  # must not raise


@pytest.mark.parametrize("allowlist,current", [
    pytest.param(["NVIDIA GeForce RTX 3090"], "NVIDIA GeForce RTX 3090", id="exact"),
    pytest.param(["RTX 3090"], "NVIDIA GeForce RTX 3090 Ti", id="substring"),
    pytest.param(["rtx 3090"], "NVIDIA GeForce RTX 3090", id="case-insensitive"),
])
def test_check_passes_for_matching_gpu(monkeypatch, allowlist, current):
    monkeypatch.setattr("pipeline.hardware._load_allowed_gpus", lambda: allowlist)
    monkeypatch.setattr("pipeline.hardware.get_current_gpu_name", lambda: current)
    check_allowed_gpu(skip=False)


def test_check_mismatch_raises_with_diagnostic(monkeypatch):
    monkeypatch.setattr("pipeline.hardware._load_allowed_gpus", lambda: ["RTX 3090"])
    monkeypatch.setattr("pipeline.hardware.get_current_gpu_name",
                        lambda: "NVIDIA H100 80GB HBM3")
    with pytest.raises(GpuMismatchError) as exc:
        check_allowed_gpu(skip=False)
    assert "H100" in str(exc.value)
    assert "allowlist" in str(exc.value).lower()

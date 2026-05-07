"""Unit tests for download_data._hub_load revision-pinning behaviour."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def _hub_load():
    from download_data import _hub_load as fn
    return fn


# ── revision present → reachability check skipped ─────────────────────────────

def test_hub_load_skips_dataset_info_when_revision_given():
    with patch("huggingface_hub.HfApi") as mock_api_cls, \
         patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = MagicMock()
        _hub_load()("owner/dataset", revision="abc123", token="tok")
        mock_api_cls.assert_not_called()


def test_hub_load_passes_revision_to_load_dataset():
    with patch("huggingface_hub.HfApi"), \
         patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = MagicMock()
        _hub_load()("owner/dataset", revision="sha1234", token="tok", split="train[:5]")
        mock_load.assert_called_once_with(
            "owner/dataset", revision="sha1234", token="tok", split="train[:5]"
        )


# ── revision absent → reachability check runs ─────────────────────────────────

def test_hub_load_calls_dataset_info_without_revision():
    with patch("huggingface_hub.HfApi") as mock_api_cls, \
         patch("datasets.load_dataset") as mock_load:
        mock_load.return_value = MagicMock()
        _hub_load()("owner/dataset", token="tok")
        mock_api_cls.assert_called_once_with(token="tok")
        mock_api_cls.return_value.dataset_info.assert_called_once_with("owner/dataset")


def test_hub_load_raises_runtime_error_when_hub_unreachable():
    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api_cls.return_value.dataset_info.side_effect = ConnectionError("network down")
        with pytest.raises(RuntimeError, match="Hugging Face Hub is unreachable"):
            _hub_load()("owner/dataset", token="tok")


def test_hub_load_wraps_any_exception_as_runtime_error():
    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api_cls.return_value.dataset_info.side_effect = TimeoutError("timed out")
        with pytest.raises(RuntimeError, match="Hugging Face Hub is unreachable"):
            _hub_load()("owner/dataset")


# ── return value ───────────────────────────────────────────────────────────────

def test_hub_load_returns_dataset_object():
    fake_ds = MagicMock(name="FakeDataset")
    with patch("huggingface_hub.HfApi"), \
         patch("datasets.load_dataset", return_value=fake_ds):
        result = _hub_load()("owner/dataset", revision="abc123")
        assert result is fake_ds

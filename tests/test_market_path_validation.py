"""
Smoke tests for market path validation infrastructure.

These tests verify the validation scripts work correctly with mock configurations.
They do NOT require actual market data files to exist.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_market_paths import load_markets_config, validate_market


class TestLoadMarketsConfig:
    """Tests for config loading functionality."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid YAML config file."""
        config_file = tmp_path / "markets.yaml"
        config_data = {
            "markets": {
                "test_market": {
                    "enabled": True,
                    "root": str(tmp_path / "data"),
                    "file_glob": "*.csv",
                }
            }
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        result = load_markets_config(config_file)

        assert "markets" in result
        assert "test_market" in result["markets"]
        assert result["markets"]["test_market"]["enabled"] is True

    def test_load_missing_config_exits(self, tmp_path: Path) -> None:
        """Test that missing config file causes exit."""
        missing_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(SystemExit) as exc_info:
            load_markets_config(missing_file)

        assert exc_info.value.code == 1


class TestValidateMarket:
    """Tests for single market validation."""

    def test_disabled_market_always_valid(self, tmp_path: Path) -> None:
        """Disabled markets should always return valid."""
        cfg = {
            "enabled": False,
            "root": str(tmp_path / "nonexistent"),
            "file_glob": "*.csv",
        }

        is_valid, details = validate_market("test", cfg)

        assert is_valid is True
        assert "disabled" in details["status"]

    def test_missing_root_is_invalid(self, tmp_path: Path) -> None:
        """Enabled market with missing root should be invalid."""
        cfg = {
            "enabled": True,
            "root": str(tmp_path / "nonexistent"),
            "file_glob": "*.csv",
        }

        is_valid, details = validate_market("test", cfg)

        assert is_valid is False
        assert "INVALID" in details["status"]
        assert "missing" in details["status"]

    def test_empty_directory_is_invalid(self, tmp_path: Path) -> None:
        """Enabled market with no matching files should be invalid."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        cfg = {
            "enabled": True,
            "root": str(data_dir),
            "file_glob": "*.csv",
        }

        is_valid, details = validate_market("test", cfg)

        assert is_valid is False
        assert "INVALID" in details["status"]
        assert "no matching files" in details["status"]

    def test_valid_market_with_files(self, tmp_path: Path) -> None:
        """Enabled market with matching files should be valid."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "BTC.csv").write_text("date,open,high,low,close,volume\n")
        (data_dir / "ETH.csv").write_text("date,open,high,low,close,volume\n")

        cfg = {
            "enabled": True,
            "root": str(data_dir),
            "file_glob": "*.csv",
        }

        is_valid, details = validate_market("test", cfg)

        assert is_valid is True
        assert details["status"] == "OK"
        assert details["file_count"] == 2
        assert len(details["sample_files"]) == 2

    def test_sample_files_limited_to_five(self, tmp_path: Path) -> None:
        """Sample files should be limited to 5."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(10):
            (data_dir / f"TOKEN{i}.csv").write_text("date,close\n")

        cfg = {
            "enabled": True,
            "root": str(data_dir),
            "file_glob": "*.csv",
        }

        is_valid, details = validate_market("test", cfg)

        assert is_valid is True
        assert details["file_count"] == 10
        assert len(details["sample_files"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from __future__ import annotations

from pathlib import Path


def optional_user_network_available(root: Path) -> tuple[bool, bool]:
    network = root / "fakehealth_zenodo" / "extracted" / "FakeHealth" / "dataset" / "user_network"
    return (network / "user_followers").exists(), (network / "user_following").exists()


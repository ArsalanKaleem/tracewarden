from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def drift_root():
    return FIX / "agentdrift"


@pytest.fixture
def trajs(drift_root):
    from tracewarden.io import agentdrift

    return agentdrift.load_part(drift_root, "train")


@pytest.fixture
def encoder():
    from tracewarden.encoders import HashingEncoder

    return HashingEncoder(64)

import copy
from pathlib import Path

import pytest

from credit_score.service import load_service


@pytest.fixture(scope="session")
def service():
    return load_service(Path(__file__).resolve().parents[1] / "artifacts")


@pytest.fixture
def profile(service):
    return copy.deepcopy(service.metadata["defaults"])

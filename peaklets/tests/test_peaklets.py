import pytest
import numpy as np
import peaklets

__all__ = [
    'test_pnpt'
]


@pytest.fixture
def signal1():
    Nt = 2048
    signal1 = np.random.rand(Nt) ** 40
    for i in range(4):
        signal1 += 0.5 * (np.roll(signal1, 1) + np.roll(signal1, -1))
    signal1 += 0.1 + 0.5 * np.random.rand(Nt)  # add some noisy background.
    return signal1


def test_pnpt(signal1: np.ndarray):
    fscales, transform, filters, pklets = peaklets.pnpt(signal1)
    assert np.all(transform >= 0)
    assert np.all(np.isclose( np.sum(transform,0), signal1 ))

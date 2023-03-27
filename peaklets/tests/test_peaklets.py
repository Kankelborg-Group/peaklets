import pytest
import numpy as np
import peaklets
from typing import Tuple

__all__ = [
    'test_pnpt'
]


@pytest.fixture
def signal1():
    Nt = 2048
    signal1 = np.random.random((Nt))
    return signal1


def test_pnpt(signal1: np.ndarray):
    scales,pklets = peaklets.pk_parabola(signal1.shape[0]) #   get scales and peaklets
    transform, filters = peaklets.pnpt(signal1, pklets, scales)
    assert np.all(transform >= 0)
    assert np.all(np.isclose( np.sum(transform,0), signal1 ))


@pytest.mark.parametrize('shape', [(128,128),(64,64,64)])
@pytest.mark.parametrize('axis',[-1,0,1]) 
def test_pkxform(shape: Tuple[int,...], axis: int):
    # help ... ???
    signal=np.random.random(shape)
    pkx = peaklets.pkxform(signal, axis)
    assert np.all(pkx.xform >= 0)
    assert np.all(np.isclose( np.sum(pkx.xform,0), signal ))

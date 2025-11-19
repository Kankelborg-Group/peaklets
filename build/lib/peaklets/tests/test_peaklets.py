import pytest
import numpy as np
import peaklets
from typing import Tuple


@pytest.fixture
def signal1():
    Nt = 2048
    signal1 = np.random.random((Nt))
    return signal1


@pytest.mark.parametrize('shape', [(128,128),(64,64,64)])
@pytest.mark.parametrize('axis',[-1,0,1]) 
def test_pkxform(shape: Tuple[int,...], axis: int):
    # help ... ???
    signal=np.random.random(shape)
    pkx = peaklets.pkxform(signal, axis)
    assert np.all(pkx.xform >= 0)
    assert np.all(np.isclose( np.sum(pkx.xform,0), signal ))

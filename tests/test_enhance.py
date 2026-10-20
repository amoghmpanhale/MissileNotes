import numpy as np

from missilenotes.enhance import binarize


def test_binarize_demo():
    gray = np.full((60, 60), 200, np.uint8)  # paper
    gray[20:40, 30:33] = 30  # a thin ink stroke
    gray[0, 0] = 20  # a 1px speck -> should die

    out = binarize(gray, block=11, C=5, min_area=6)

    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert out[30, 31] == 0, "ink should be black"
    assert out[0, 0] == 255, "speck should be removed"
    assert out[5, 55] == 255, "paper should be white"
    assert set(np.unique(out)) <= {0, 255}

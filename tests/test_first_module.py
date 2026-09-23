import math

import numpy as np
import pytest

from MLGWSC_proposal.first_module import find_perp, rot_angle, rot_axis


def test_rot_angle() -> None:
    assert math.isclose(rot_angle(np.array([0, 1, 0]), np.array([1, 0, 0])), np.pi / 2)

    assert math.isclose(rot_angle(np.array([1, 0, 0]), np.array([1, 0, 0])), 0)

    with pytest.raises(ValueError):
        rot_angle(np.array([0, 0, 0]), np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        rot_angle(np.array([1, 2, 3]), np.array([0, 0, 0]))


def test_rot_axis() -> None:
    assert np.allclose(
        rot_axis(np.array([0, 1, 0]), -np.pi / 2, np.array([0, 0, 1])), np.array([1, 0, 0])
    )

    assert np.allclose(rot_axis(np.array([1, 2, 3]), 0, np.array([0, 0, 1])), np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        rot_axis(np.array([0, 0, 0]), -np.pi / 2, np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        rot_axis(np.array([1, 2, 3]), -np.pi / 2, np.array([0, 0, 0]))


def test_find_perp() -> None:
    assert np.allclose(find_perp(np.array([1, 0, 0]), np.array([0, 1, 0])), np.array([0, 0, 1]))

    assert np.allclose(find_perp(np.array([1, 2, 0]), np.array([3, 4, 0])), np.array([0, 0, -1]))

    with pytest.raises(ValueError):
        find_perp(np.array([0, 0, 0]), np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        find_perp(np.array([1, 2, 3]), np.array([0, 0, 0]))

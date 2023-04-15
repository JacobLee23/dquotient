"""
Unit tests for :py:mod:`dquotient`.
"""

from decimal import Decimal
import pytest

from dquotient import FiniteDifference
from dquotient import FunctionRV


@pytest.mark.parametrize(
    ("func", "x", "h", "fdiffs"),
    [
        (lambda _: 0, 0, 0, FiniteDifference.FDifferences(0, 0, 0)),
    ]
)
def test_finitedifference(
    func: FunctionRV, x: Decimal, h: Decimal, fdiffs: FiniteDifference.FDifferences
    ):
    """
    Unit tests for :py:class:`dquotient.FiniteDifference`.
    """
    assert FiniteDifference.fdifferences(func, x, h) == fdiffs

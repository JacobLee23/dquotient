"""
Unit tests for :py:mod:`dquotient`.
"""

from decimal import Decimal
import pytest

from dquotient import DifferenceQuotient as DQuot
from dquotient import FiniteDifference as FDiff
from dquotient import FunctionRV


@pytest.mark.parametrize(
    ("func", "x", "h", "fdiffs"),
    [
        (lambda _: 0, 0, 0, FDiff.FDifferences(0, 0, 0)),
    ]
)
def test_finitedifference(
    func: FunctionRV, x: Decimal, h: Decimal, fdiffs: FDiff.FDifferences
):
    """
    Unit tests for :py:class:`dquotient.FiniteDifference`.
    """
    assert FDiff.fdifferences(func, x, h) == fdiffs


@pytest.mark.parametrize(
    ("func", "x", "nderiv"),
    [
        (lambda _: 0, 0, 0),
    ]
)
def test_differencequotient(
    func: FunctionRV, x: Decimal, nderiv: DQuot.NDeriv
):
    """
    Unit tests for :py:class:`dquotient.DifferenceQuotient`.
    """
    diff = DQuot(func)
    assert diff.nderiv(x) == nderiv

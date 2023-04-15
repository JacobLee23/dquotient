"""
Computation of finite differences and the difference quotient.
"""

import decimal
from decimal import Decimal
import typing


FunctionRV = typing.Callable[[Decimal], Decimal]
FDiffMethod = typing.Callable[[FunctionRV, Decimal, Decimal], Decimal]


class FiniteDifference:
    r"""
    .. math::

        f(x + b) - f(x + a)
    """
    class FDifferences(typing.NamedTuple):
        """
        .. py:attribute:: forward

        .. py:attribute:: backward

        .. py:attribute:: central
        """
        forward: Decimal
        backward: Decimal
        central: Decimal

    @staticmethod
    def forward(func: FunctionRV, x: Decimal, h: Decimal, *, prec: int = 100) -> Decimal:
        r"""
        Computes the forward difference of ``func``:
        
        .. math::
        
            {\Delta}_{h}[f](x) = f(x + h) - f(x)
            
        :param func:
        :param x:
        :param h:
        :param prec:
        :return: The forward difference of ``func`` at ``x``
        """
        with decimal.localcontext() as ctx:
            ctx.prec = 2 * prec

            return (func(x + h) - func(x))

    @staticmethod
    def backward(func: FunctionRV, x: Decimal, h: Decimal, *, prec: int = 100) -> Decimal:
        r"""
        Computes the backward difference of ``func``:
        
        .. math::
        
            {\nabla}_{h}[f](x) = f(x) - f(x - h)
            
        :param func:
        :param x:
        :param h:
        :param prec:
        :return: The backward difference of ``func`` at ``x``
        """
        with decimal.localcontext() as ctx:
            ctx.prec = 2 * prec

            return func(x) - func(x - h)

    @staticmethod
    def central(func: FunctionRV, x: Decimal, h: Decimal, *, prec: int = 100) -> Decimal:
        r"""
        Computes the central difference of ``func``:
        
        .. math::
        
            {\deta}_{h}[f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})
            
        :param func:
        :param x:
        :param h:
        :param prec:
        :return: The central difference of ``func`` at ``x``
        """
        with decimal.localcontext() as ctx:
            ctx.prec = 2 * prec

            return func(x + h / 2) - func(x - h / 2)

    @classmethod
    def fdifferences(cls, func: FunctionRV, x: Decimal, h: Decimal, *, prec: int = 100) -> FDifferences:
        r"""
        :param func:
        :param x:
        :param h:
        :param prec:
        :return:
        """
        return cls.FDifferences(
            cls.forward(func, x, h, prec=prec),
            cls.backward(func, x, h, prec=prec),
            cls.central(func, x, h, prec=prec)
        )


class DifferenceQuotient:
    r"""
    .. math::

        {f}^{'}(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
    """
    class NDeriv(typing.NamedTuple):
        """
        .. py:attribute:: left

        .. py:attribute:: central

        .. py:attribute:: right
        """
        left: Decimal
        central: Decimal
        right: Decimal

    def __init__(self, func: FunctionRV):
        self.func = func

    def nderiv(self, x: Decimal, *, prec: int = 100) -> NDeriv:
        r"""
        :param fdiff:
        :param x:
        :param prec:
        :return:
        """
        with decimal.localcontext() as ctx:
            ctx.prec = prec + 2

            h = Decimal(f"1E-{prec}")

            left = FiniteDifference.backward(self.func, x, h) / h
            central = FiniteDifference.central(self.func, x, h) / h
            right = FiniteDifference.forward(self.func, x, h) / h

            return self.NDeriv(left, central, right)

    def derivative(self, *, prec: int = 100) -> typing.Callable[[Decimal], NDeriv]:
        r"""
        :param fdiff:
        :param prec:
        :return:
        """
        def _derivative(x: Decimal) -> self.NDeriv:
            r"""
            :param x:
            :return:
            """
            return self.nderiv(x, prec=prec)

        return _derivative

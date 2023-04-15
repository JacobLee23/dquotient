"""
"""

import decimal
from decimal import Decimal
from numbers import Number
import typing


OneVarFunction = typing.Callable[[Number], Number]
FDiffType = typing.Callable[[OneVarFunction, Decimal, Decimal], Decimal]


class FiniteDifference:
    """
    """
    @staticmethod
    def forward(func: OneVarFunction, x: Decimal, h: Decimal) -> Decimal:
        r"""
        Computes the forward difference of ``func``:
        
        .. math::
        
            {\Delta}_{h}[f](x) = f(x + h) - f(x)
            
        :param x:
        :param h:
        :return: The forward difference of ``func`` at ``x``
        """
        return func(x + h) - func(x)
    
    @staticmethod
    def backward(func: OneVarFunction, x: Decimal, h: Decimal) -> Decimal:
        r"""
        Computes the backward difference of ``func``:
        
        .. math::
        
            {\nabla}_{h}[f](x) = f(x) - f(x - h)
            
        :param x:
        :param h:
        :return: The backward difference of ``func`` at ``x``
        """
        return func(x) - func(x - h)
    
    @staticmethod
    def central(func: OneVarFunction, x: Decimal, h: Decimal) -> Decimal:
        r"""
        Computes the central difference of ``func``:
        
        .. math::
        
            {\deta}_{h}[f](x) = f(x + \frac{h}{2}) - f(x - \frac{h}{2})
            
        :param x:
        :param h:
        :return: The central difference of ``func`` at ``x``
        """
        return func(x + h / 2) - func(x - h / 2)

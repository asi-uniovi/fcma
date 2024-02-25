from pint import DimensionalityError, UnitRegistry
from pint.facets.plain import PlainQuantity as Quantity
from pint.testing import assert_allclose as assert_approx
from typing import Union, cast

# Define new units
ureg = UnitRegistry()
ureg.define("usd = [currency]")
ureg.define("core = [computation] = 2*vcores")
ureg.define("millicore = 0.001 cores")
ureg.define("req = [requests]")
# ureg.define("bit = [storage]")
ureg.define("[performance] = [requests]/[time]")
ureg.define("rps = [performance] = req/second")
ureg.define("rpm = req/minute")
ureg.define("rph = req/hour")


class CheckedDimensionality(Quantity):
    _my_dimensionality = "[]"

    def __init_subclass__(cls, dimensionality="[]", **kwargs):
        # This function is called when this class is subclassed
        # We store the dimensionality value in the subclass
        super().__init_subclass__(**kwargs)
        cls._my_dimensionality = dimensionality

    def __new__(cls, v: Union[str, Quantity]) -> "CheckedDimensionality":
        # This method will be inherited by the subclasses and used to
        # create obejects of that subclass. During the creation,
        # the correct dimensionality is checked
        obj = ureg.Quantity(v)  # type: ignore
        if not obj.check(cls._my_dimensionality):
            raise DimensionalityError(v, cls._my_dimensionality)
        return obj


# -----------------------------------------------------------------------------------
# New dimension types appropriate for the cloud model
# -----------------------------------------------------------------------------------
class Time(CheckedDimensionality, dimensionality="[time]"):
    def to(self, other=None, *ctx, **ctx_kwargs):  # pragma: no cover
        return cast("Time", super().to(other, *ctx, **ctx_kwargs))


class Storage(CheckedDimensionality, dimensionality="[]"):
    def to(self, other=None, *ctx, **ctx_kwargs):  # pragma: no cover
        return cast("Storage", super().to(other, *ctx, **ctx_kwargs))


class RequestsPerTime(CheckedDimensionality, dimensionality="[requests]/[time]"):
    def to(self, other=None, *ctx, **ctx_kwargs):  # pragma: no cover
        return cast("RequestsPerTime", super().to(other, *ctx, **ctx_kwargs))


class CurrencyPerTime(CheckedDimensionality, dimensionality="[currency]/[time]"):
    def to(self, other=None, *ctx, **ctx_kwargs):  # pragma: no cover
        return cast("CurrencyPerTime", super().to(other, *ctx, **ctx_kwargs))


class ComputationalUnits(CheckedDimensionality, dimensionality="[computation]"):
    def to(self, other=None, *ctx, **ctx_kwargs):  # pragma: no cover
        return cast("ComputationalUnits", super().to(other, *ctx, **ctx_kwargs))


class Currency(CheckedDimensionality, dimensionality="[currency]"):
    ...


class Requests(CheckedDimensionality, dimensionality="[requests]"):
    ...


UNLIMITED_TIME = Time("1 year")

__all__ = [
    "Time",
    "Currency",
    "CurrencyPerTime",
    "Requests",
    "RequestsPerTime",
    "Storage",
    "ComputationalUnits",
    "UNLIMITED_TIME",
]

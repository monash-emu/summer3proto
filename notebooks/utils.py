from jax import numpy as jnp
import numpy as np
from proto import CompartmentContainer, strats_for_cmap
from datetime import datetime, timedelta
import pandas as pd
from numbers import Real


def _interp_from_ranges(t, x, y, xiranges, yranges):
    xidx = jnp.searchsorted(x, t)
    xl = jnp.where(xidx > 0, x[xidx - 1], x[0])
    yout = y[xidx - 1] + (t - xl) * xiranges[xidx] * yranges[xidx]
    yout = jnp.where(xidx >= len(x), y[-1], yout)
    yout = jnp.where(xidx < 1, y[0], yout)
    return yout


class LinearInterpolator:
    def __init__(self, x, y):
        self.xiranges = 1.0 / jnp.diff(x, prepend=x[0], append=x[-1])
        self.yranges = jnp.diff(y, prepend=y[0], append=y[-1])
        self.x = x
        self.y = y

    def process(self, t):
        return _interp_from_ranges(t, self.x, self.y, self.xiranges, self.yranges)


def get_strat_prop_dicts(cmap: CompartmentContainer):
    strat_prop_dicts = {
        strat.name: np.empty(len(cmap), dtype=object) for strat in strats_for_cmap(cmap)
    }
    for i, c in enumerate(cmap.compartments):
        for strat, stratum in c.strata:
            strat_prop_dicts[strat.name][i] = stratum

    return strat_prop_dicts


class Epoch:
    """Epoch converts between numeric offset values (eg model times), and concrete datetime values"""

    def __init__(self, ref_date: datetime, unit: timedelta = timedelta(1)):
        """Create an Epoch for conversion. ref_date will be equivalent to 0.0,
        with other dates offset in units of unit

        Args:
            ref_date: The '0-time' reference data
            unit (optional): timedelta unit equivalent to 1.0 steps in numeric representation
        """
        self.ref_date = ref_date
        self.unit = unit

    def __repr__(self):
        return f"Epoch from {self.ref_date}, in units of {self.unit}"

    def index_to_dti(self, index: pd.Index) -> pd.DatetimeIndex:
        """Convert an index (or iterable) of numeric values to a DatetimeIndex

        Args:
            index (pd.Index): Inputs to convert

        Returns:
            Equivalent DatetimeIndex
        """
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        return self.ref_date + index * self.unit

    def dti_to_index(self, index: pd.DatetimeIndex) -> pd.Index:
        """Convert a DatetimeIndex to its equivalent numeric representation

        Args:
            index: The input index to convert

        Returns:
            Equivalent numerical index
        """
        return (index - self.ref_date) / self.unit

    def number_to_datetime(self, n: Real) -> datetime:
        """Convert a single number to a datetime

        Args:
            n: The number

        Returns:
            The datetime
        """
        return self.ref_date + n * self.unit

    def datetime_to_number(self, d: datetime) -> float:
        """Convert a datetime number to a float

        Args:
            d: The datetime

        Returns:
            The number
        """
        return (d - self.ref_date) / self.unit


def get_category_names(cat_groups):
    return [
        "_".join(["|".join([stratum for stratum in strata]) for strat, strata in cat])
        for cat in cat_groups
    ]

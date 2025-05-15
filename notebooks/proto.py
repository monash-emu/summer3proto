import numpy as np
import itertools
from typing import Optional
from warnings import warn
from copy import deepcopy

import jax
from jax import numpy as jnp


class Stratification:
    def __init__(self, name: str, strata: list[str]):
        self.name = name
        self.strata = strata

    def __repr__(self):
        return f"Stratification: {self.name}"

    # +++
    # Provide an easy way to obtain StratSpec - maybe getitem?


class Compartment:
    def __init__(self, strata: list[tuple[Stratification, str]], index: int):
        self.strata = strata
        self.index = index

    def __repr__(self):
        return "Compartment :" + repr(self.strata)


StratSpec = tuple[Stratification, str] | tuple[Stratification, list[str]] | None
StratMap = dict[Stratification, StratSpec]
CompartmentArray = np.ndarray[Compartment]


class CompartmentContainer:
    compartments: CompartmentArray

    def __init__(self, compartments: CompartmentArray):
        self.compartments = np.array(compartments)

    def query(self, traits: list[StratSpec]):
        qres = []
        indices = []
        for i, c in enumerate(self.compartments):
            has_all = True
            for t in traits:
                has_trait = False
                for sspec in iter_stratspec(t):
                    has_trait = has_trait or sspec in c.strata
                if not has_trait:
                    has_all = False
                    break
            if has_all:
                qres.append(c)
                indices.append(i)
        return CompartmentView(np.array(qres), self, np.array(indices))


class CompartmentMap(CompartmentContainer):
    def __init__(self, compartments: CompartmentArray, stratifications: StratMap):
        super().__init__(compartments)

        self.stratifications = stratifications
        self._base_strat = list(stratifications)[0]

    @classmethod
    def new(cls, base_strat: Stratification):
        compartments = np.array(
            [Compartment([(base_strat, s)], i) for i, s in enumerate(base_strat.strata)]
        )
        stratifications: dict[Stratification, Optional[tuple]] = {base_strat: None}
        return cls(compartments, stratifications)

    def stratify(
        self, strat: Stratification, stratifies: StratSpec = None, in_place=True
    ):

        if stratifies is None:
            stratifies = (self._base_strat, self._base_strat.strata)
        if isinstance(stratifies[1], str):
            stratifies = (stratifies[0], [stratifies[1]])

        target_strat, target_strata = stratifies

        for existing_strat, estrat_stratifies in self.stratifications.items():
            if existing_strat.name == strat.name:
                warn(f"Existing stratification with name {strat.name}")
                # +++ Actually check for overlap, not just equivalency
                if estrat_stratifies == stratifies:
                    raise Exception(
                        "Existing stratification with same name overlaps",
                        strat.name,
                        stratifies,
                    )
        out_comps = []
        new_comps = []
        i = 0

        remapped_comps = {}

        for c in self.compartments:
            if any(
                [(target_strat, t_stratum) in c.strata for t_stratum in target_strata]
            ):
                remapped_comps[c] = []
                for stratum in strat.strata:
                    new_c = Compartment(c.strata + [(strat, stratum)], i)
                    out_comps.append(new_c)
                    new_comps.append(new_c)
                    remapped_comps[c].append(new_c)
                    i += 1
            else:
                out_comps.append(c)
                remapped_comps[c] = [c]
                i += 1

        if len(new_comps) == 0:
            raise Exception("No compartments match stratification request", stratifies)

        if in_place:
            self.compartments = np.array(out_comps)
            self.stratifications[strat] = stratifies
        else:
            stratifications = self.stratifications.copy()
            stratifications[strat] = stratifies
            return CompartmentMap(out_comps, stratifications), strat, remapped_comps
        return self, strat, remapped_comps

    def rebase(
        self, new_base_strat: Stratification, key, in_place=False
    ) -> "CompartmentMap":
        new_stratifications = {new_base_strat: None}

        for k, v in self.stratifications.items():
            if k is self._base_strat:
                new_stratifications[k] = (k, [key])
            else:
                new_stratifications[k] = v
        # self.stratifications = new_stratifications
        new_c = [
            Compartment([(new_base_strat, key)] + c.strata, c.index)
            for c in self.compartments
        ]

        if in_place:
            self.stratifications = new_stratifications
            self._base_strat = new_base_strat
            self.compartments = new_c
            return self
        else:
            return CompartmentMap(new_c, new_stratifications)

    def add_compartments(self, other_comps: "CompartmentMap", base_stratum: str):
        idx = len(self.compartments)
        assert not any(
            [other_s in self.stratifications for other_s in other_comps.stratifications]
        )

        if base_stratum not in self._base_strat.strata:
            raise KeyError("Stratum not found in base stratification", base_stratum)

        other_rebased = other_comps.rebase(self._base_strat, base_stratum)
        for c in other_comps.compartments:

            new_c = Compartment([(self._base_strat, base_stratum)] + c.strata, idx)
            self.compartments.append(new_c)
            idx += 1

        for k, v in other_rebased.stratifications.items():
            self.stratifications[k] = v


class CompartmentView(CompartmentContainer):
    """View of a subset of compartments"""

    def __init__(
        self,
        compartments: CompartmentArray,
        parent: CompartmentMap,
        indices: np.ndarray,
    ):
        super().__init__(compartments)
        self.parent = parent
        self.indices = indices

    def __repr__(self):
        return (
            f"CompartmentView of {self.parent}:\n"
            + repr(self.compartments)
            + repr(self.indices)
        )


def iter_stratspec(sspec: StratSpec):
    strat, strata = sspec
    for stratum in strata:
        yield (strat, stratum)


class CompartmentGroup:
    """Maybe need a better name - really just a container of traits (arguments to a query)"""

    def __init__(self, traits=list[StratSpec]):
        self.traits = traits

    def query(self, cmap: CompartmentMap):
        qres = []
        indices = []
        for i, c in enumerate(cmap.compartments):
            has_all = True
            for t in self.traits:
                has_trait = False
                for sspec in iter_stratspec(t):
                    has_trait = has_trait or sspec in c.strata
                if not has_trait:
                    has_all = False
                    break
            if has_all:
                qres.append(c)
                indices.append(i)
        return CompartmentView(np.array(qres), cmap, np.array(indices))


def category_idx_reduction(cat_indices: list[np.ndarray], src: jax.Array):
    if len(set([len(c) for c in cat_indices])) == 1:
        return src[np.array(cat_indices)].sum(axis=1)
    else:
        return jnp.array([src[c].sum() for c in cat_indices])


class LA:
    def __init__(self, data, axes):
        self.data = data
        self.axes = axes

        self._ax_to_idx = {k: i for i, k in enumerate(axes)}
        self._idx_to_ax = {v: k for k, v in self._ax_to_idx.items()}

    def transpose(self, axes):
        transposed = self.data.transpose([self._ax_to_idx[a] for a in axes])
        return LA(transposed, axes)

    def expand(self, ax, index=-1):
        expanded = jnp.expand_dims(self.data, index)
        if index == -1:
            new_axes = self.axes + [ax]
        else:
            new_axes = []
            for a, i in enumerate(self.axes):
                if i == index:
                    new_axes.append(ax)
                new_axes.append(a)
        return LA(expanded, new_axes)

    def reconcile(self, other):
        s_set = set(self.axes)
        o_set = set(other.axes)
        s_extras = s_set.difference(o_set)
        o_extras = o_set.difference(s_set)
        other_expanded = other
        self_expanded = self
        for eax in list(s_extras):
            other_expanded = other_expanded.expand(eax)
        for eax in list(o_extras):
            self_expanded = self_expanded.expand(eax)
        return self_expanded, other_expanded.transpose(self_expanded.axes)

    def to_axis(self, ax):
        if ax not in self.axes:
            raise KeyError("Axis not found", ax)
        return [a for a in self.axes if a != ax]

    def __mul__(self, other):
        return self._lop(other, jnp.multiply)

    def __rmul__(self, other):
        return self._rop(other, jnp.multiply)

    def __truediv__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data / orec.data, srec.axes)

    def __rtruediv__(self, other):
        return self._rop(other, jnp.true_divide)

    def _lop(self, other, op):
        if isinstance(other, float):
            return LA(op(self.data, other), self.axes)
        srec, orec = self.reconcile(other)
        return LA(op(srec.data, orec.data), srec.axes)

    def _rop(self, other, op):
        if isinstance(other, float):
            return LA(op(other, self.data), self.axes)
        else:
            raise TypeError("Unsupported type", other)

    def __add__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data + orec.data, srec.axes)

    def __sub__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data - orec.data, srec.axes)

    def sum(self, axis=None):
        return self._liftreduction("sum", axis)

    def _reduce_axes(self, axis=None):
        if axis is None:
            lifted_ax = None
            reduced_axes = self.axes
        elif isinstance(axis, str):
            lifted_ax = self._ax_to_idx[axis]
            reduced_axes = [a for a in self.axes if a != axis]
        else:
            lifted_ax = [self._ax_to_idx[a] for a in axis]
            reduced_axes = [a for a in self.axes if a not in axis]
        return lifted_ax, reduced_axes

    def _liftreduction(self, op, axis=None):
        lifted_ax, reduced_axes = self._reduce_axes(axis)
        return LA(getattr(self.data, op)(axis=lifted_ax), reduced_axes)

    def __repr__(self):
        data_repr = repr(self.data)
        info_repr = f"LA {self.axes} {self.data.shape}\n"
        return info_repr + data_repr

import numpy as np
import itertools
from typing import Optional
from warnings import warn
from copy import deepcopy

import jax
from jax import numpy as jnp
import numpy as np


class Stratification:
    def __init__(self, name: str, strata: list[str]):
        self.name = name
        self.strata = strata

    def __repr__(self):
        return f"Stratification: {self.name}"

    def __getitem__(self, k):
        if isinstance(k, str):
            if k in self.strata:
                return (self, [k])
            else:
                raise KeyError()
        elif k is Ellipsis:
            return (self, [strat for strat in self.strata])
        else:
            strata = [ki for ki in k]
            for s in strata:
                if s not in self.strata:
                    raise KeyError()
            return (self, [ki for ki in k])

    def categories(self):
        return [[(self, [stratum])] for stratum in self.strata]

    # +++
    # Provide an easy way to obtain StratSpec - maybe getitem?


class Compartment:
    def __init__(self, strata: list[tuple[Stratification, str]]):
        self.strata = strata

    def __repr__(self):
        return "Compartment :" + repr(self.strata)

    # def __eq__(self, other):
    #    return set(self.strata) == set(other.strata)

    def __hash__(self):
        return hash(tuple(*(self.strata,)))

    def matches(self, other, strats):
        a_strats = {strat: stratum for (strat, stratum) in self.strata}
        b_strats = {strat: stratum for (strat, stratum) in other.strata}
        for strat in strats:
            if a_strats[strat] != b_strats[strat]:
                return False
        return True


StratSpec = tuple[Stratification, str] | tuple[Stratification, list[str]] | None
StratMap = dict[Stratification, StratSpec]
CompartmentArray = np.ndarray[Compartment]


def validate_qspec(qspec: list[StratSpec]):
    if isinstance(qspec, list):
        return qspec
    elif isinstance(qspec, tuple):
        return [qspec]
    raise TypeError("Invalid query specification")


class CompartmentContainer:
    compartments: CompartmentArray

    def __init__(
        self,
        compartments: CompartmentArray,
        root: "CompartmentMap" = None,
        parent: "CompartmentContainer" = None,
        indices: np.array = None,
    ):
        self.compartments = np.array(compartments)
        if parent is None and indices is None:
            parent = self
            indices = np.arange(len(compartments))

        if parent is not None and indices is not None:
            self.parent = parent
            self.indices = indices
        else:
            raise Exception("Both or neither of parent and indices must be specified")

        self.root = root

    def __getitem__(self, indices):
        compartments = self.compartments[indices]
        return CompartmentContainer(compartments, self.root, self, indices)

    def query(self, traits: list[StratSpec]) -> "CompartmentContainer":
        traits = validate_qspec(traits)
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
        return CompartmentContainer(np.array(qres), self.root, self, np.array(indices))

    def wrap_data(self, data):
        return CompartmentDataContainer(self.compartments, self, data)

    def zeros(self, lib=jnp):
        return CompartmentDataContainer(
            self.compartments, self, lib.zeros(len(self.compartments))
        )

    def __repr__(self):
        if self.parent == self:
            return "CompartmentContainer:\n" + repr(self.compartments)
        else:
            return (
                f"CompartmentContainer view of 0x{id(self.parent)}:\n"
                + repr(self.compartments)
                + repr(self.indices)
            )

    def __len__(self):
        return len(self.compartments)


class CompartmentMap(CompartmentContainer):
    def __init__(self, compartments: CompartmentArray, stratifications: StratMap):
        super().__init__(compartments)

        self.root = self
        self.stratifications = stratifications
        self._base_strat = list(stratifications)[0]
        self.remappings = {}

    @classmethod
    def new(cls, base_strat: Stratification):
        compartments = np.array(
            [Compartment([(base_strat, s)]) for s in base_strat.strata]
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
                    new_c = Compartment(c.strata + [(strat, stratum)])
                    out_comps.append(new_c)
                    new_comps.append(new_c)
                    remapped_comps[c].append(new_c)
                    i += 1
            else:
                out_comps.append(c)
                i += 1

        if len(new_comps) == 0:
            raise Exception("No compartments match stratification request", stratifies)

        if in_place:
            self.compartments = np.array(out_comps)
            self.stratifications[strat] = stratifies
            self.remappings[strat] = remapped_comps
        else:
            stratifications = self.stratifications.copy()
            stratifications[strat] = stratifies
            new_cmap = CompartmentMap(out_comps, stratifications)
            new_cmap.remappings = self.remappings.copy()
            new_cmap.remappings[strat] = remapped_comps
            return new_cmap, strat
        return strat

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
            Compartment([(new_base_strat, key)] + c.strata) for c in self.compartments
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


class CompartmentDataContainer(CompartmentContainer):
    def __init__(
        self,
        compartments: CompartmentArray,
        root: CompartmentMap,
        data: jnp.array,
        parent: "CompartmentDataContainer" = None,
        indices: np.array = None,
    ):
        super().__init__(
            compartments=compartments, root=root, parent=parent, indices=indices
        )
        self.data = data

    def query(self, traits: list[StratSpec]):
        qcomp_view = super().query(traits)
        return CompartmentDataContainer(
            qcomp_view.compartments,
            self.root,
            self.data[qcomp_view.indices],
            self,
            qcomp_view.indices,
        )

    def __repr__(self):
        if self.parent == self:
            return (
                "CompartmentDataContainer:\n"
                + repr(self.compartments)
                + repr(self.data)
            )
        else:
            return (
                f"CompartmentDataContainer view of 0x{id(self.parent)}:\n"
                f"Compartments:\n{repr(self.compartments)}\n"
                f"Indices:\n{repr(self.indices)}\n"
                f"Data:\n{repr(self.data)}\n"
            )


def iter_stratspec(sspec: StratSpec):
    strat, strata = sspec
    for stratum in strata:
        yield (strat, stratum)


def category_idx_reduction(cat_indices: list[np.ndarray], src: jax.Array):
    if len(set([len(c) for c in cat_indices])) == 1:
        return src[np.array(cat_indices)].sum(axis=1)
    else:
        return jnp.array([src[c].sum() for c in cat_indices])


def query_cat_reduction(query_cats, comp_data):
    indices = [comp_data.query(qc).indices for qc in query_cats]
    return category_idx_reduction(indices, comp_data.data)


def cat_indices(query_cats, comp_data):
    indices = [comp_data.query(qc).indices for qc in query_cats]
    return np.array(indices)


### Flows
def strats_for_comp(c):
    strats = []
    for strat, stratum in c.strata:
        strats.append(strat)
    return list(set(strats))


def strats_for_cmap(cmap):
    src_strats = set()
    for c in cmap.compartments:
        cstrats = strats_for_comp(c)
        for s in cstrats:
            src_strats.add(s)
    return list(src_strats)


def reconcile_broadcast(srcq, destq, cmap, strategy=None):
    src = cmap.query(srcq)
    dest = cmap.query(destq)
    if len(src.compartments) == len(dest.compartments):
        return src, dest, None
    else:
        if len(src) > len(dest):
            print("Gather")
            src_tmp = src
            src = dest
            dest = src_tmp
            scatter = False
        else:
            print("Scatter")
            scatter = True
        src_strats = set(strats_for_cmap(src))
        dest_strats = set(strats_for_cmap(dest))
        transition_strats = set([strat for (strat, q) in srcq])
        common_strats = src_strats.intersection(dest_strats) - transition_strats
        scatters = list(dest_strats.difference(src_strats))

        comp_idx = {c: i for i, c in enumerate(cmap.compartments)}

        if len(scatters):
            scatter_strat = scatters[0]
            out_src_comps = []
            out_dest_comps = []
            out_src_indices = []
            out_dest_indices = []
            adj = []

            for src_comp in src.compartments:
                for dest_comp in dest.compartments:
                    if src_comp.matches(dest_comp, common_strats):
                        out_src_comps.append(src_comp)
                        out_dest_comps.append(dest_comp)
                        out_src_indices.append(comp_idx[src_comp])
                        out_dest_indices.append(comp_idx[dest_comp])
                        if scatter:
                            adj.append(1.0 / len(scatter_strat.strata))

            out_src_comps = np.array(out_src_comps)
            out_dest_comps = np.array(out_dest_comps)
            out_src_indices = np.array(out_src_indices)
            out_dest_indices = np.array(out_dest_indices)

            rec_src = CompartmentContainer(
                out_src_comps, src.root, src.root, out_src_indices
            )
            rec_dest = CompartmentContainer(
                out_dest_comps, dest.root, dest.root, out_dest_indices
            )
            if scatter:
                return rec_src, rec_dest, np.array(adj)
            else:
                return rec_dest, rec_src, None


class CategoryData:
    def __init__(self, cats, data):
        self.cats = cats
        self.data = data

    def __repr__(self):
        return f"CategoryData:\n{self.cats}\n{self.data}\n"


class ActualizedTransitionFlow:
    def __init__(self, flow, src_cmap, dest_cmap, adjustments, apply_func):
        self.flow = flow
        self.src_cmap = src_cmap
        self.dest_cmap = dest_cmap
        self.adjustments = adjustments
        self.get_flow_vals = apply_func


class TransitionFlow:
    def __init__(self, srcq, destq, param):
        self.srcq = validate_qspec(srcq)
        self.destq = validate_qspec(destq)
        self.param = param
        self.adjustments = []

    def actualize(self, cmap, param_key=None):

        param_key = param_key or self.param

        realised_adjustments = []

        src_cmap, dest_cmap, adj = reconcile_broadcast(self.srcq, self.destq, cmap)

        if adj is not None:
            realised_adjustments.append(adj)

        def apply_flow(cdatamap, params):
            src_comp_vals = cdatamap.data[src_cmap.indices]
            param = params[param_key]
            if isinstance(param, CategoryData):
                cidx = cat_indices(param.cats, src_cmap)
                flow_vals = src_comp_vals.at[cidx.T].mul(param.data)
            else:
                flow_vals = param * src_comp_vals
            for adj in realised_adjustments:
                flow_vals = flow_vals * adj
            return flow_vals

        return ActualizedTransitionFlow(
            self, src_cmap, dest_cmap, realised_adjustments, apply_flow
        )


class ActualizedExitFlow:
    def __init__(self, flow, src_cmap, adjustments, apply_func):
        self.flow = flow
        self.src_cmap = src_cmap
        self.adjustments = adjustments
        self.get_flow_vals = apply_func


class ExitFlow:
    def __init__(self, srcq, param):
        self.srcq = validate_qspec(srcq)
        self.param = param
        self.adjustments = []

    def actualize(self, cmap):
        src_cmap = cmap.query(self.srcq)

        def apply_flow(cdatamap, params):
            src_comp_vals = cdatamap.data[src_cmap.indices]
            param = params[self.param]
            if isinstance(param, CategoryData):
                cidx = cat_indices(param.cats, src_cmap)
                flow_vals = src_comp_vals.at[cidx.T].mul(param.data)
            else:
                flow_vals = params[self.param] * src_comp_vals
            return flow_vals

        return ActualizedExitFlow(self, src_cmap, self.adjustments, apply_flow)


class ActualizedEntryFlow:
    def __init__(self, flow, dest_cmap, adjustments, apply_func):
        self.flow = flow
        self.dest_cmap = dest_cmap
        self.adjustments = adjustments
        self.get_flow_vals = apply_func


class EntryFlow:
    def __init__(self, destq, param):
        self.destq = validate_qspec(destq)
        self.param = param
        self.adjustments = []

    def actualize(self, cmap):
        dest_cmap = cmap.query(self.destq)

        def apply_flow(cdatamap, params):
            param = params[self.param]
            if isinstance(param, CategoryData):
                raise Exception("CategoryData not yet supported for EntryFlow")
            else:
                flow_vals = param
            return flow_vals

        return ActualizedEntryFlow(self, dest_cmap, self.adjustments, apply_flow)

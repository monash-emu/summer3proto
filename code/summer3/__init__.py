import numpy as np
from jax import numpy as jnp

# from summer2 import inspect as mi

from . import inspect as mi


class NStrat:
    def __init__(self, name, strata, stratifies=None, is_base=False):
        self.name = name
        self.strata = strata
        self.is_base = is_base
        self.stratifies = stratifies or {}

    def __repr__(self):
        return f"{self.name}: {self.strata}"


class NComp:
    def __init__(self, name, strata, idx=None):
        self.name = name
        self.strata = strata
        self.idx = idx

    def __repr__(self):
        return f"NComp<{self.name}>"

    def __hash__(self) -> int:
        return self.name.__hash__()


class NFlow:
    def __init__(self, name, src_comps, dest_comps):
        self.name = name
        self.src_comps = []
        self.dest_comps = []

    def __repr__(self):
        return f"NFlow<{self.name}>"


class CompartmentQuery:
    def __init__(self, data: list[NComp]):
        self.compartments = data

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.compartments]

    @property
    def index(self) -> np.ndarray[int]:
        return np.array([c.idx for c in self.compartments], dtype=int)

    def __repr__(self):
        return f"CompartmentQuery: {self.compartments.__repr__()}"


class NModel:
    def __init__(self, init_comps, init_strat="state"):
        self.compartments = [
            NComp(k, {init_strat: k}, i) for i, k in enumerate(init_comps)
        ]
        self.flows = []
        self.stratifications = {init_strat: NStrat(init_strat, init_comps, True)}

    def query_compartments(self, q: dict = None) -> CompartmentQuery:
        q = q or {}
        return CompartmentQuery(mi.query_compartments(self, q))

    def stratify(self, strat):
        assert strat.name not in self.stratifications

        comps_to_stratify = self.query_compartments(strat.stratifies).compartments

        source_comps = comps_to_stratify
        dest_comps = []
        new_comps = []

        for c in self.compartments:
            if c in comps_to_stratify:
                # for f in self.flows:
                #     if c in f.src_comps:

                #     if c in f.dest_comps:

                cur_new = [
                    NComp("_".join((c.name, stratum)), c.strata | {strat.name: stratum})
                    for stratum in strat.strata
                ]
                new_comps += cur_new
                dest_comps += cur_new
            else:
                new_comps.append(c)

        for i, c in enumerate(new_comps):
            c.idx = i

        self.compartments = new_comps
        self.stratifications[strat.name] = strat

        # self._transactions.append()


def get_category_indexer(m: NModel, query: list[dict]):
    return np.array([m.query_compartments(q).index for q in query])


def get_category_counts(m: NModel, query: list[dict], compartment_values):
    query_vals = [m.query_compartments(q).index for q in query]
    base_len = len(query_vals[0])
    if all([len(q) == base_len for q in query_vals[1:]]):
        return compartment_values[np.array(query_vals)].sum(axis=1)
    else:
        return jnp.array([compartment_values[q].sum() for q in query_vals])


def proportional(weights):
    return weights / weights.sum()

"""Tools for probing, querying, inspecting and drawing CompartmentalModels

The main entry point for this is the ModelProbe class
"""

import re
from typing import Iterable, Set, List, Callable

import itertools

import networkx as nx
import numpy as np

# from summer2 import CompartmentalModel
# from summer2.compartment import Compartment
# from summer2.flows import BaseFlow


def _query_compartments(m, query: dict = None, tags: List = None, as_idx=False):
    query = query or {}
    tags = tags or []
    if isinstance(tags, str):
        tags = [tags]
    if "name" in query:
        query = query.copy()
        name = query.pop("name")
        matching = m.get_matching_compartments(name, query)
        if as_idx:
            return np.array(
                [c.idx for c in matching if all([t in c.tags for t in tags])], dtype=int
            )
        else:
            return [c for c in matching if all([t in c.tags for t in tags])]
    else:
        _strata = frozenset(query.items())
        if as_idx:
            return np.array(
                [
                    c.idx
                    for c in m.compartments
                    if c._has_strata(_strata) and all([t in c.tags for t in tags])
                ],
                dtype=int,
            )
        else:
            return [
                c
                for c in m.compartments
                if c._has_strata(_strata) and all([t in c.tags for t in tags])
            ]


def query_compartments(model, query: dict, tags: list = None, as_idx=False):
    query = query or {}

    tags = tags or []
    if isinstance(tags, str):
        tags = [tags]

    if "name" in query:
        query = query.copy()
        name = query.pop("name")

        if isinstance(name, str):
            compartments = model._compartment_name_map[name]
        elif isinstance(name, Callable):
            match_lists = [
                model._compartment_name_map[n]
                for n in model._original_compartment_names
                if name(n)
            ]
            compartments = list(itertools.chain.from_iterable(match_lists))
        elif isinstance(name, Iterable):
            # FIXME: Should do better type checking here
            # For now we assume we have some kind of iterable (ie a 'list' of names)
            match_lists = [model._compartment_name_map[n] for n in name]
            compartments = list(itertools.chain.from_iterable(match_lists))
        else:
            raise TypeError()
    else:
        compartments = model.compartments

    def get_equals(x):
        def equals(y):
            return y == x

        return equals

    def get_isin(x):
        def isin(y):
            return y in x

        return isin

    actual_q = {}
    for k, v in query.items():
        if isinstance(v, Callable):
            actual_q[k] = v
        elif isinstance(v, str):
            actual_q[k] = get_equals(v)
        elif isinstance(v, Iterable):
            actual_q[k] = get_isin(v)
        else:
            raise TypeError()

    matched_comps = []
    for c in compartments:
        cur_match = True
        for stratification, qfunc in actual_q.items():
            if stratification in c.strata:
                cur_match = cur_match and qfunc(c.strata[stratification])
            else:
                cur_match = False

        if cur_match:
            matched_comps.append(c)

    if len(tags):
        matched_comps = [c for c in matched_comps if all([t in c.tags for t in tags])]

    if as_idx:
        return np.array([c.idx for c in matched_comps], dtype=int)
    else:
        return matched_comps


def query_flows(
    m,
    flow_name: str = None,
    source: dict = None,
    dest: dict = None,
    tags: List = None,
):
    if flow_name is not None:
        if isinstance(flow_name, re.Pattern):
            flows = [f for f in m.flows if flow_name.match(f.name)]
        elif isinstance(flow_name, str):
            flows = [f for f in m.flows if flow_name == f.name]
        else:
            flows = flow_name
    else:
        flows = m.flows

    if source:
        if "name" in source:
            source = source.copy()
            name = source.pop("name")
            flows = [f for f in flows if f.source and f.source.name == name]
        else:
            source = frozenset(source.items())
            flows = [f for f in flows if f.source and f.source._has_strata(source)]

    if dest:
        if "name" in dest:
            dest = dest.copy()
            name = dest.pop("name")
            flows = [f for f in flows if f.dest and f.dest.name == name]
        else:
            dest = frozenset(dest.items())
            flows = [f for f in flows if f.dest and f.dest._has_strata(source)]

    if tags:
        if isinstance(tags, str):
            tags = [tags]
        flows = [f for f in flows if all([t in f.tags for t in tags])]

    return flows


def flows_to_compartments(m, flows):
    comps = []
    for f in flows:
        if f.source:
            comps.append(f.source)
        if f.dest:
            comps.append(f.dest)
    return set(comps)


def build_compartment_flow_map(m):
    out_map = {c: set() for c in m.compartments}
    for f in m.flows:
        if f.source:
            out_map[f.source].add(f)
        if f.dest:
            out_map[f.dest].add(f)
    return out_map


"""
Tools specific to handling models as networkx graph structures
"""


def model_to_digraph(compartments, flows):
    g = nx.DiGraph()
    for c in compartments:
        g.add_node(c)

    def is_fully_mapped(f, comps):
        return f.source in comps and f.dest in comps

    for f in flows:
        if is_fully_mapped(f, compartments):
            g.add_edge(str(f.source), str(f.dest))

    return g

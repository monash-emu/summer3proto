import polars as pl
import numpy as np
from bidict import bidict

from typing import Sequence

from summer3.polarized.properties import Property, PropertyTable, concat_pt
from summer3.polarized.flows import FlowSpec, source, dest


def get_age_state_severity_pt():
    age = Property("age", ["infant", "child", "adult", "older"])
    state = Property("state", ["S", "I", "R"])
    severity = Property("severity", ["mild", "severe"])

    pt = (
        PropertyTable.from_property(state)
        .stratify(severity, state == "I")
        .stratify(age, state)
    )

    return age, state, severity, pt


def test_all_to_one():
    age, state, severity, pt = get_age_state_severity_pt()
    all_to_infant = FlowSpec(age, (age == "infant") & (state == "S"), pt)
    fspt = all_to_infant.get_flow_pt()
    assert (fspt.df["index_dest"] == 0).all()
    assert (fspt.df["index_source"].to_numpy() == np.arange(16)).all()


def test_one_to_many():
    age, state, severity, pt = get_age_state_severity_pt()


def test_single_strat_move():
    age, state, severity, pt = get_age_state_severity_pt()
    infection = FlowSpec(state == "S", state == "I", pt)
    assert set(infection.get_flow_pt().df["index_dest"]) == set(
        pt.filter(state == "I").df["index"]
    )
    assert set(infection.get_flow_pt().df["index_source"]) == set(
        pt.filter(state == "S").df["index"]
    )
    # Source is broadcast over severity
    assert (
        infection.get_flow_pt().df["index_source"].value_counts()["count"]
        == len(severity)
    ).all()


def test_cats_concat_equivalent():
    age, state, severity, pt = get_age_state_severity_pt()
    pta = FlowSpec(age == "infant", age == "child", pt).get_flow_pt()
    ptb = FlowSpec(age == "child", age == "adult", pt).get_flow_pt()
    ptc = FlowSpec(age == "adult", age == "older", pt).get_flow_pt()

    assert concat_pt([pta, ptb, ptc]).df.equals(
        FlowSpec(age[:-1].categories(), age[1:].categories(), pt).get_flow_pt().df
    )

import pytest
from build import BuildGraph, CycleError

def valid(order, edges):
    pos = {t: i for i, t in enumerate(order)}
    return all(pos[d] < pos[t] for t, ds in edges.items() for d in ds)

def test_simple_chain():
    g = BuildGraph()
    g.add_target("app", ["lib"])
    g.add_target("lib", ["core"])
    assert g.build_order() == ["core", "lib", "app"]

def test_dep_only_names_included_and_alpha_ties():
    g = BuildGraph()
    g.add_target("z", ["b", "a"])
    g.add_target("y", [])
    assert g.build_order() == ["a", "b", "y", "z"]

def test_diamond():
    g = BuildGraph()
    edges = {"d": ["b", "c"], "b": ["a"], "c": ["a"], "a": []}
    for t, ds in edges.items():
        g.add_target(t, ds)
    order = g.build_order()
    assert order == ["a", "b", "c", "d"]
    assert valid(order, edges)

def test_readd_replaces_deps():
    g = BuildGraph()
    g.add_target("x", ["y"])
    g.add_target("x", [])
    g.add_target("y", ["x"])           # no cycle: x no longer depends on y
    assert g.build_order() == ["x", "y"]

def test_cycle_raises():
    g = BuildGraph()
    g.add_target("a", ["b"])
    g.add_target("b", ["c"])
    with pytest.raises(CycleError):    # Level 1: build_order raises. Level 3: add_target raises.
        g.add_target("c", ["a"])
        g.build_order()

def test_self_loop_raises():
    g = BuildGraph()
    with pytest.raises(CycleError):
        g.add_target("a", ["a"])
        g.build_order()

def test_empty():
    assert BuildGraph().build_order() == []
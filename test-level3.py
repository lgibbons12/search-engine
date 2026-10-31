import pytest
from build import BuildGraph, CycleError

def test_reject_cycle_and_leave_graph_unchanged():
    g = BuildGraph()
    g.add_target("a", ["b"])
    g.add_target("b", ["c"])
    with pytest.raises(CycleError):
        g.add_target("c", ["a"])
    assert g.build_order() == ["c", "b", "a"]
    assert g.deps_of("c") == []

def test_rejected_replace_keeps_old_deps():
    g = BuildGraph()
    g.add_target("a", ["b"])
    g.add_target("x", ["y"])
    with pytest.raises(CycleError):
        g.add_target("b", ["x", "a"])   # replacing b's deps with ones that cycle
    assert g.deps_of("a") == ["b"]
    assert g.deps_of("b") == []
    assert g.affected_by("y") == ["y", "x"]

def test_self_loop_rejected():
    g = BuildGraph()
    with pytest.raises(CycleError):
        g.add_target("a", ["a"])
    assert g.build_order() == []

def test_remove_target():
    g = BuildGraph()
    g.add_target("app", ["lib"])
    g.add_target("lib", [])
    assert g.remove_target("lib") is False      # app depends on it
    assert g.remove_target("app") is True
    assert g.remove_target("app") is False      # already gone
    assert g.build_order() == ["lib"]
    assert g.remove_target("lib") is True
    assert g.build_order() == []

def test_remove_keeps_deps():
    g = BuildGraph()
    g.add_target("app", ["lib"])
    assert g.remove_target("app") is True
    assert g.build_order() == ["lib"]
    assert g.affected_by("lib") == ["lib"]
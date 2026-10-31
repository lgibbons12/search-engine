from build import BuildGraph

def test_default_times_is_longest_chain():
    g = BuildGraph()
    g.add_target("c", ["b"]); g.add_target("b", ["a"]); g.add_target("x", [])
    assert g.min_build_time() == 3
    assert g.critical_path() == ["a", "b", "c"]

def test_weighted_diamond():
    g = BuildGraph()
    g.add_target("d", ["b", "c"]); g.add_target("b", ["a"]); g.add_target("c", ["a"])
    for t, s in {"a": 2, "b": 5, "c": 1, "d": 3}.items():
        g.set_time(t, s)
    assert g.min_build_time() == 10
    assert g.critical_path() == ["a", "b", "d"]

def test_parallel_independent():
    g = BuildGraph()
    g.add_target("p", []); g.add_target("q", [])
    g.set_time("p", 4); g.set_time("q", 7)
    assert g.min_build_time() == 7
    assert g.critical_path() == ["q"]

def test_long_cheap_chain_vs_short_expensive():
    g = BuildGraph()
    g.add_target("a3", ["a2"]); g.add_target("a2", ["a1"])
    g.add_target("big", [])
    g.set_time("big", 10)
    assert g.min_build_time() == 10
    assert g.critical_path() == ["big"]

def test_empty():
    g = BuildGraph()
    assert g.min_build_time() == 0
    assert g.critical_path() == []

def test_large_chain_is_fast():
    g = BuildGraph()
    n = 3000
    for i in range(1, n):
        g.add_target(f"t{i:05d}", [f"t{i-1:05d}"])
    assert g.min_build_time() == n
    assert len(g.critical_path()) == n
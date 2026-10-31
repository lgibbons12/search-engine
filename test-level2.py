from build import BuildGraph

def make():
    g = BuildGraph()
    g.add_target("app", ["ui", "net"])
    g.add_target("ui", ["core"])
    g.add_target("net", ["core", "ssl"])
    g.add_target("cli", ["core"])
    return g

def test_deps_of():
    g = make()
    assert g.deps_of("app") == ["core", "net", "ssl", "ui"]
    assert g.deps_of("ui") == ["core"]
    assert g.deps_of("core") == []
    assert g.deps_of("nope") == []

def test_affected_by_in_build_order():
    g = make()
    assert g.affected_by("core") == ["core", "cli", "net", "ui", "app"]
    assert g.affected_by("ssl") == ["ssl", "net", "app"]
    assert g.affected_by("app") == ["app"]
    assert g.affected_by("nope") == ["nope"]

def test_shared_dep_counted_once():
    g = BuildGraph()
    g.add_target("d", ["b", "c"]); g.add_target("b", ["a"]); g.add_target("c", ["a"])
    assert g.deps_of("d") == ["a", "b", "c"]
    assert g.affected_by("a") == ["a", "b", "c", "d"]
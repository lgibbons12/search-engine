"""
Part 1: data structure is dependency dictionary: dict[item] -> things that depend on it

add_target(name, deps) sets adds name as something that depends on all the dependencies in deps
- dictionary hash is O(1) expecting and set add is O(1) and we do that #deps times so O(#deps) expected 

build_order() does a DFS that uses a heap to choose alphabetically next dependency based on all that are free.
- DFS is O(#targets + #dependency arrows) but we must maintain a heap which each item is added to and 
  removed from once so O(# targets log (# targets)) so total is O(# target log(# targets) + #dependency arrows)
  We also store a set of what has been added to the heap so if we see something twice we raise CycleError


Part 2: reuse dependency dictionary
deps_of(name) does something similar to build order just walking the dictionary of things it depends on. Unknown name -> []
- this is just O(#dependencies log (#dependnecies)) as we walk the dependencies once but then sort at the end 

affected_by(name) does the opposite and looks at everything it goes before. Unknown name -> [name]
- this is just O(# affected by log(# affected by)) as everything must go in and out of the heap once

Part 3:
- `add_target` must now **reject** an edit that would create a cycle: raise `CycleError`
  and leave the graph **unchanged** --> just do the build order check in add target

Did not get to:
- `remove_target(name) -> bool`: remove `name` only if nothing depends on it. Returns `True` if
  removed, `False` if something depends on it or it doesn't exist.
  (Removing `name` doesn't remove its deps.)

Part 4
"""
from collections import defaultdict
import heapq

class CycleError(Exception):
    pass


class BuildGraph:
    def __init__(self):
        self.goes_before = defaultdict(set)
        self.depends_on = dict()
        self.sources = set()

    def add_target(self, name: str, deps: list[str]) -> None:
        if name in set(deps):
            raise CycleError
        num_source_adds = len([dep for dep in deps if dep not in self.depends_on])
        if (1 if not deps else 0) + (-1 if name in self.sources else 0) + num_source_adds < 0:
            raise CycleError

        deps_to_remove = self.depends_on.get(name, set())

        for dep in deps_to_remove:
            self.goes_before[dep].remove(name)

        self.depends_on[name] = set(deps)
        
        if not deps:
            self.sources.add(name)
            return
        if name in self.sources:
            self.sources.remove(name)
        
        for dep in deps:
            if dep not in self.depends_on:
                self.sources.add(dep)
            self.goes_before[dep].add(name)

    def build_order(self) -> list[str]:
        if not self.sources and self.goes_before:
            raise CycleError
        heap = [name for name in self.sources]
        heapq.heapify(heap)
        depend_count = defaultdict(int)
        out = []
        while heap:
            node = heapq.heappop(heap)
            if node in self.goes_before:
                for next in self.goes_before[node]:                  
                    depend_count[next] += 1
                    if depend_count[next] == len(self.depends_on[next]):
                        heapq.heappush(heap, next)

            out.append(node)

        return out
    


    def affected_by(self, name: str) -> list[str]:
        heap = [name]
        seen = set()
        out = []
        while heap:
            curr_heap = []
            while heap: 
                node = heapq.heappop(heap)
                if node in self.goes_before:
                    for next in self.goes_before[node]:                  
                        if next not in seen:
                            heapq.heappush(curr_heap, next)
                            seen.add(next)
                out.append(node)

            heap = curr_heap

        return out

    def deps_of(self, name: str) -> list[str]:
        if name not in self.depends_on:
            return []
        stack = [next for next in self.depends_on[name]]
        seen = set([next for next in stack])
        out = []
        while stack:
            node = stack.pop()
            if node in self.depends_on:
                for next in self.depends_on[node]:
                    if next not in seen:
                        stack.append(next)
                        seen.add(next)
            out.append(node)

        return sorted(out)
            


if __name__ == "__main__":
    # x = BuildGraph()
    # x.add_target('a', ['b', 'c'])
    # assert x.sources == set(['b', 'c'])
    # assert x.build_order() == ['b', 'c', 'a']
    # x.add_target('b', ['e'])
    # assert x.sources == set(['e', 'c'])
    # assert x.goes_before == {'b': set(['a']), 'c': set(['a']), 'e': set(['b'])}
    # assert x.depends_on == {'a': set(['b', 'c']), 'b': set(['e'])}
    # assert x.build_order() == ['c', 'e', 'b', 'a']

    # g = BuildGraph()
    # g.add_target("x", ["y"])
    # g.add_target("x", [])
    # g.add_target("y", ["x"])           # no cycle: x no longer depends on y
    # assert g.build_order() == ["x", "y"]

    # g = BuildGraph()
    # g.add_target("a", ["b"])
    # g.add_target("b", ["c"])
    # print(f"{g.goes_before=}, {g.depends_on=}, {g.sources=}")
    # g.add_target("c", ["a"])
    # print(f"{g.goes_before=}, {g.depends_on=}, {g.sources=}")
    # g.build_order()

    # g = BuildGraph()
    # g.add_target("app", ["ui", "net"])
    # g.add_target("ui", ["core"])
    # g.add_target("net", ["core", "ssl"])
    # g.add_target("cli", ["core"])
    # print(f"{g.goes_before=}, {g.depends_on=}")
    # print(g.deps_of("app"))

    g = BuildGraph()
    g.add_target("a", ["b"])
    g.add_target("b", ["c"])
    print(f"{g.goes_before=}, {g.depends_on=}, {g.sources=}")

    g.add_target("c", ["a"])
    print(f"{g.goes_before=}, {g.depends_on=}, {g.sources=}")
    assert g.build_order() == ["c", "b", "a"]
    assert g.deps_of("c") == []
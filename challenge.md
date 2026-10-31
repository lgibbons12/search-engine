# Coding challenge: build system (75 min, solo, no AI)

Implement `BuildGraph` in `build.py`. Four levels; each builds on the last.
Run `pytest -q` (all) or `pytest -q test_level1.py`. Dependency chains can be thousands of targets long.

A **target** (string) can depend on other targets: you must build its dependencies before it.
Before each level, write 2–3 lines at the top of `build.py`: your data structures, and the
complexity of each method. (This stands in for aligning out loud.)

## Level 1: targets and build order
- `add_target(name, deps) -> None`: register `name` with a list of dependency names. A dependency
  may be added before or after the target that uses it. Re-adding a name **replaces** its deps.
- `build_order() -> list[str]`: every target that has been added, plus every name that appears
  only as a dependency, in an order where each target comes after all of its deps.
  Ties are broken **alphabetically** (when several targets are ready, the smallest name goes first).
  If there's a cycle, raise `CycleError` (defined for you).

## Level 2: dependency queries
- `deps_of(name) -> list[str]`: all **transitive** dependencies of `name` (not including `name`),
  sorted alphabetically. Unknown name → `[]`.
- `affected_by(name) -> list[str]`: everything that must be rebuilt if `name` changes:
  `name` itself plus everything that transitively depends on it, **in a valid build order**
  (same tie-break as Level 1). Unknown name → `[name]`.

## Level 3: safe edits
- `add_target` must now **reject** an edit that would create a cycle: raise `CycleError`
  and leave the graph **unchanged**.
- `remove_target(name) -> bool`: remove `name` only if nothing depends on it. Returns `True` if
  removed, `False` if something depends on it or it doesn't exist.
  (Removing `name` doesn't remove its deps.)

## Level 4: parallel builds
Each target now has a build time.
- `set_time(name, t) -> None`: build time in seconds (default 1 for targets without one).
- `min_build_time() -> int`: with unlimited parallel workers, the minimum wall-clock time to
  build everything. (A target can start once all its deps have finished.) Empty graph → 0.
- `critical_path() -> list[str]`: one longest chain of targets that determines that time, from
  first-built to last-built. Ties: any valid answer of maximum length is accepted.

## Level 5: review (last 10 minutes, no matter what)
In `REVIEW.md`:
1. The time complexity of each method you implemented, in V and E.
2. What structure is this input, and which guarantee (if any) lets Level 4 be linear?
3. One bug you'd look for in someone else's Level 3, plus a test in `test_mine.py` that catches it.
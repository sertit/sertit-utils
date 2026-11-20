"""
Check which sertit modules import, and how guarded functions fail.

CI installs all the optional dependencies, so a missing one is
simulated: the checks run in a fresh subprocess where a sys.meta_path
finder makes the selected imports raise ModuleNotFoundError. It has
to be a subprocess -- once a module is imported, the current
process's import cache can't be rolled back.

Nothing is hardcoded: modules come from pkgutil, guarded functions
from the _sertit_requires marker set by utils.requires.

For a quick report:

    python ci/import_check.py rasterio,click
    python ci/import_check.py            # nothing blocked
"""

import importlib
import os
import pkgutil
import subprocess
import sys
from pathlib import Path

import sertit


def modules() -> list[str]:
    """Public modules of the sertit package (discovered with pkgutil)."""
    return [
        m.name
        for m in pkgutil.iter_modules(sertit.__path__)
        if not m.name.startswith("_")
    ]


def guarded_calls(module_name: str):
    """Yield (dotted_name, deps, function) for the guarded functions of a module.

    Each guarded function is reported once, under the module that
    defines it (imported names are recognized via __module__,
    duplicates via id).
    """
    module = importlib.import_module(f"sertit.{module_name}")
    seen: set[int] = set()
    for name in dir(module):
        fn = getattr(module, name)
        deps = getattr(fn, "_sertit_requires", None)
        # not a tuple: not a guarded function (some callables, like
        # lxml's ElementMaker, synthesize an attribute for any name)
        if not isinstance(deps, tuple) or id(fn) in seen:
            continue
        if getattr(fn, "__module__", None) != f"sertit.{module_name}":
            continue
        seen.add(id(fn))
        yield f"{module_name}.{name}", list(deps), fn


class _BlockFinder:
    """Makes the given imports raise ModuleNotFoundError, like a real missing module.

    Raising (rather than returning None) also breaks transitive
    imports: blocking geopandas breaks `import geoutils` too, which
    is what we want -- the guard then reports geoutils, as it would
    if geoutils were genuinely absent.
    """

    def __init__(self, blocked: set[str]) -> None:
        self.blocked = blocked

    def find_spec(self, name, path=None, target=None):
        if name in self.blocked or any(name.startswith(f"{b}.") for b in self.blocked):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return None


def _expected_guard(deps: list[tuple[str, str]]) -> str:
    """Which guard should fire first ("module|extra"), or "OK".

    Same import attempts, in the same order, as utils.requires.
    """
    for module, extra in deps:
        try:
            importlib.import_module(module)
        except ImportError:
            return f"{module}|{extra}"
    return "OK"


def check(
    blocked: tuple[str, ...] = (),
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Run the checks in a subprocess where `blocked` imports fail.

    The subprocess is pointed at the package this module imported
    (via PYTHONPATH), so a stale installed copy of sertit is never
    tested by accident.

    Returns (imports, calls): module -> result line, and
    "module.function" -> (expected guard, result line).
    """
    pythonpath = [str(Path(sertit.__path__[0]).parent)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath)}
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), ",".join(blocked)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert proc.returncode == 0, f"import check crashed:\n{proc.stderr}"
    imports: dict[str, str] = {}
    calls: dict[str, tuple[str, str]] = {}
    for line in proc.stdout.splitlines():
        fields = line.split("\t")
        if fields[0].startswith("import:"):
            imports[fields[0].removeprefix("import:")] = fields[1]
        elif fields[0].startswith("call:"):
            calls[fields[0].removeprefix("call:")] = (fields[1], fields[2])
    return imports, calls


def main(blocked: set[str]) -> None:
    """Import every sertit module and call every guarded function."""
    sys.meta_path.insert(0, _BlockFinder(blocked))

    for module in modules():
        try:
            importlib.import_module(f"sertit.{module}")
        except BaseException as ex:
            print(f"import:{module}\t{type(ex).__name__}: {ex}")
        else:
            print(f"import:{module}\tOK")

    for module in modules():
        try:
            importlib.import_module(f"sertit.{module}")
        except BaseException:
            continue
        for dotted, deps, fn in guarded_calls(module):
            expected = _expected_guard(deps)
            try:
                fn()
            except BaseException as ex:
                print(f"call:{dotted}\t{expected}\t{type(ex).__name__}: {ex}")
            else:
                print(f"call:{dotted}\t{expected}\tOK")


if __name__ == "__main__":
    names = sys.argv[1] if len(sys.argv) > 1 else ""
    main({name for name in names.split(",") if name})

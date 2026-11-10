"""``import escape`` loads its public names and subpackages on first use
(PEP 562): fast, and everything reachable as before."""

import subprocess
import sys


def _run(code):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_import_escape_loads_no_data_stack():
    assert _run("import escape, sys; print('dask' in sys.modules or 'h5py' in sys.modules)") == "False"


def test_public_names_and_subpackages_as_attributes():
    code = (
        "import escape\n"
        "escape.Array, escape.DataSet, escape.concatenate\n"
        "for name in ('storage', 'utilities', 'adaptive_scan', 'plot_utilities', 'stream', 'swissfel'):\n"
        "    getattr(escape, name)\n"
        "from escape import Array, DataSet, adaptive_scan\n"
        "print('ok')"
    )
    assert _run(code) == "ok"


def test_unknown_names_are_attribute_errors():
    code = (
        "import escape\n"
        "try:\n"
        "    escape.no_such_thing\n"
        "except AttributeError as e:\n"
        "    print('AttributeError')\n"
        "print(hasattr(escape, 'no_such_thing'), 'importlib' in dir(escape))"
    )
    assert _run(code).splitlines() == ["AttributeError", "False False"]

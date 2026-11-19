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


def test_swissfel_imports_from_a_worker_thread():
    """``escape.swissfel`` must not install a signal handler while importing.

    ``dask_jobqueue``'s runner module calls ``signal.signal(SIGINT, ...)`` at
    module scope, which CPython allows only on the main thread. While that
    import sat at the top of ``swissfel/cluster.py``, the first thread to
    import ``escape.swissfel`` decided whether it worked at all: a caller
    whose first import happened on a worker thread got

        ValueError: signal only works in main thread of the main interpreter

    and, because a failed import is not cached, got it again on every retry.
    That cost a control system two of its devices, which built correctly in
    an interactive session and never in its thread pool.
    """
    code = (
        "import threading\n"
        "res = []\n"
        "def imp():\n"
        "    try:\n"
        "        import escape.swissfel\n"
        "        res.append('ok')\n"
        "    except Exception as exc:\n"
        "        res.append(f'{type(exc).__name__}: {exc}')\n"
        "t = threading.Thread(target=imp); t.start(); t.join()\n"
        "print(res[0])"
    )
    assert _run(code) == "ok"


def test_importing_swissfel_does_not_pull_in_dask_jobqueue():
    """Only SwissFelCluster(local=False) needs SLURM, so nothing else should
    pay for it - and keeping it out is what keeps the test above true."""
    code = (
        "import sys, escape.swissfel\n"
        "print('dask_jobqueue' in sys.modules)"
    )
    assert _run(code) == "False"

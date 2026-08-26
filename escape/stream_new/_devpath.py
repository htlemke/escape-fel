"""Make example notebooks import the `escape` checkout they live in.

Jupyter starts a kernel with its cwd set to the notebook's own directory, and that
directory is on sys.path by default -- so `import _devpath` works with no path setup,
regardless of where this checkout is located on disk. Call
`prepend_local_checkout_to_path()` before `import escape` to make sure the notebook
uses this checkout instead of any escape-fel install found elsewhere on sys.path.
"""

import sys
from pathlib import Path


def prepend_local_checkout_to_path(package="escape", start=None):
    """Find the checkout of `package` containing this file and put it first on sys.path.

    Walks upward from `start` (default: this file's directory) until it finds a
    folder with a `<package>/__init__.py` in it, then inserts that folder at the
    front of sys.path. If `package` was already imported (e.g. from an installed
    copy), it's dropped from sys.modules first so the next `import package` re-reads
    from this checkout.
    """
    start = Path(start) if start is not None else Path(__file__).resolve().parent
    for d in (start, *start.parents):
        if (d / package / "__init__.py").is_file():
            d_str = str(d)
            if d_str in sys.path:
                sys.path.remove(d_str)
            sys.path.insert(0, d_str)
            for mod in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
                del sys.modules[mod]
            return d
    raise FileNotFoundError(f"no '{package}' package found above {start}")

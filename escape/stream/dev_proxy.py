"""
Dev-only helper: route escape.stream's live-data traffic through a local
SSH SOCKS proxy.

Not part of escape.stream's public API (see CLAUDE.md) -- this is a debugging
convenience for developing escape.stream itself against *real* SwissFEL
channels from a machine that is not on the PSI network, given a SOCKS-mode
SSH tunnel into one that is, e.g.::

    ssh -D 8787 -N your_user@saresb-cons-05

Then, before creating any EventWorker/DataHubEventHandler::

    from escape.stream.dev_proxy import enable_socks_proxy
    enable_socks_proxy(port=8787)

    from escape.stream import EventWorker, DataHubEventHandler
    ew = EventWorker(DataHubEventHandler(backend='bsread'))
    ew.registerSource('SAROP21-PBPS103:INTENSITY')

Two independent legs get routed, since bsread/datahub speak two different
wire protocols to two different PSI hosts:

- **The dispatcher's plain HTTP(S) lookup** (``datahub`` resolves a channel
  list to a data-stream URL via ``requests.get/post``) picks up the standard
  ``HTTP_PROXY``/``HTTPS_PROXY`` environment variables automatically --
  ``requests`` (and the urllib3 it sits on) already honors those with no code
  changes. ``socks5h://`` (not ``socks5://``) is used deliberately so that
  hostname resolution also happens through the tunnel: the dispatcher and
  data hosts (``dispatcher-api.psi.ch``, ``sf-daqbuf-NN``, ...) are internal
  PSI names that won't resolve from off-network. This requires the optional
  ``PySocks`` dependency (``pip install pysocks``, or the ``requests[socks]``
  extra).
- **The actual bsread/mflow data socket** is raw ZMQ, which bypasses
  ``requests``/Python's ``socket`` module entirely (libzmq does its own
  connecting in C), so environment-variable proxying does not reach it.
  ``pyzmq`` does support a native per-socket ``SOCKS_PROXY`` option (pyzmq
  >= ~19, libzmq >= 4.1). ``mflow`` (which bsread/datahub sit on) creates its
  own ``zmq.Context``/socket internally with no proxy hook exposed, so the
  only seam available is ``zmq.Context.socket()`` itself -- patching it means
  every socket *any* zmq-using code in this process creates afterwards gets
  the option set before it connects.

Caveat: once enabled, this also redirects connections meant for a *local*
synthetic stream (``TestStream``/``LocalEventHandler``) through the tunnel --
"localhost" would then mean the tunnel endpoint's loopback, not this
machine's. Call :func:`disable_socks_proxy` before going back to local-only
testing with ``TestStream``.
"""

import os

_PROXY_ENV_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")

_saved_env = {}
_orig_context_socket = None
_proxy_bytes = None


def enable_socks_proxy(host="127.0.0.1", port=8787):
    """Route escape.stream's HTTP + ZMQ traffic through a local SOCKS5 proxy.

    Safe to call more than once -- a later call updates the target (env vars
    and the ZMQ proxy option both pick up the new *host*/*port*) without
    double-patching.

    Parameters
    ----------
    host : str
        Local SOCKS5 proxy address, e.g. the ``ssh -D`` bind address.
    port : int
        Local SOCKS5 proxy port.

    Raises
    ------
    RuntimeError
        If ``pyzmq`` is missing, or too old to have the ``SOCKS_PROXY``
        socket option (the HTTP/dispatcher leg would still get proxied via
        the environment variables either way, but the actual data socket
        would silently bypass the tunnel, which is worse than failing loudly).
    """
    global _orig_context_socket, _proxy_bytes

    proxy_url = f"socks5h://{host}:{port}"
    for var in _PROXY_ENV_VARS:
        _saved_env.setdefault(var, os.environ.get(var))
        os.environ[var] = proxy_url

    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required to route the bsread/mflow data socket through "
            "a SOCKS proxy."
        ) from exc
    if not hasattr(zmq, "SOCKS_PROXY"):
        raise RuntimeError(
            "This pyzmq build has no SOCKS_PROXY socket option (need "
            "pyzmq >= ~19 / libzmq >= 4.1). The HTTP/dispatcher leg is still "
            "routed via HTTP_PROXY/HTTPS_PROXY, but the bsread data socket "
            "itself will bypass the tunnel and fail to connect off-network."
        )

    _proxy_bytes = f"{host}:{port}".encode()

    if _orig_context_socket is None:
        _orig_context_socket = zmq.Context.socket

        def _socket_via_socks(self, *args, **kwargs):
            sock = _orig_context_socket(self, *args, **kwargs)
            sock.setsockopt(zmq.SOCKS_PROXY, _proxy_bytes)
            return sock

        zmq.Context.socket = _socket_via_socks

    print(f"escape.stream.dev_proxy: routing HTTP + ZMQ traffic through socks5h://{host}:{port}")


def disable_socks_proxy():
    """Undo enable_socks_proxy(): restore the environment and un-patch zmq."""
    global _orig_context_socket, _proxy_bytes

    for var, val in _saved_env.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val
    _saved_env.clear()

    if _orig_context_socket is not None:
        import zmq
        zmq.Context.socket = _orig_context_socket
        _orig_context_socket = None
    _proxy_bytes = None

    print("escape.stream.dev_proxy: SOCKS proxy routing disabled.")

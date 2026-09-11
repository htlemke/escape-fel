"""
bsread event-source wrappers.

Improvements over the original es_wrappers.py:
  - EventSourceContext (renamed from EventSource) handles a None bsread source
    gracefully so that the EventWorker can start before any channels are registered.
  - LocalEventHandler connects directly to a host:port without going through the
    SwissFEL dispatcher — use this with the local test stream.
  - EventHandler_SFEL no longer hard-codes the dispatcher URL; it falls back to
    whatever bsread uses as its own default (currently sf-databuffer).  The old URL
    '/sf' is stale and causes connection failures.
  - NullEvent is returned when no source is configured, preventing KeyErrors in the
    event-loop callbacks while still providing a valid (but empty) event interface.
"""

import time


# ---------------------------------------------------------------------------
# Null / sentinel event
# ---------------------------------------------------------------------------

class NullEvent:
    """Placeholder event emitted when no bsread source is connected.

    All data lookups return None; the event-ID is None so that EscData
    de-duplication filters it out before any data is appended.
    """

    def getFromSource(self, source):  # noqa: N802
        return None

    def getEventId(self):  # noqa: N802
        return None

    def get_channel_names(self):  # noqa: N802
        return []


# ---------------------------------------------------------------------------
# Context manager that wraps a live bsread connection
# ---------------------------------------------------------------------------

class EventSourceContext:
    """Context manager that owns a connected bsread Source.

    Handles the case where ``eventhandler.source`` is None (no channels
    registered yet) by sleeping and returning NullEvents.
    """

    def __init__(self, eventhandler):
        self.eventhandler = eventhandler

    def __enter__(self):
        src = self.eventhandler.source
        if src is not None:
            src.connect()
        return self

    def get_event(self):
        src = self.eventhandler.source
        if src is None:
            time.sleep(0.05)
            return NullEvent()
        msg = src.receive()
        if msg is None:
            return NullEvent()
        return Event_SFEL(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        src = self.eventhandler.source
        if src is not None:
            src.disconnect()
        return False


# ---------------------------------------------------------------------------
# Event wrapper
# ---------------------------------------------------------------------------

class Event_SFEL:
    """Wraps a raw bsread message into the escape event interface."""

    def __init__(self, message):
        self.message = message

    def getFromSource(self, source):  # noqa: N802
        if source == "lab_time":
            ts = self.message.data.global_timestamp
            ts_ns = self.message.data.global_timestamp_offset
            return ts + 1e-9 * ts_ns
        if source == "pulse_id":
            return self.message.data.pulse_id
        return self.message.data.data[source].value

    def getEventId(self):  # noqa: N802
        return self.message.data.pulse_id

    def get_channel_names(self):  # noqa: N802
        return list(self.message.data.data.keys())


# ---------------------------------------------------------------------------
# SwissFEL event handler (uses dispatcher)
# ---------------------------------------------------------------------------

class EventHandler_SFEL:
    """EventHandler for live SwissFEL bsread streams via the PSI dispatcher.

    The dispatcher URL is intentionally left as bsread's own default so that
    this code does not need to be updated when PSI changes the URL.  Pass an
    explicit ``dispatcher_url`` kwarg in ``source_kwargs`` to override.
    """

    def __init__(self, **source_kwargs):
        # Only set keys that differ from bsread's own defaults.
        self._source_kwargs = dict(
            host=None,
            port=9999,
            queue_size=100,
            copy=True,
            all_channels=False,
            receive_timeout=None,
        )
        self._source_kwargs.update(source_kwargs)
        self.source = None
        self.source_ids = []

    def clone(self):
        """Return a fresh, independent handler with the same configuration
        (no channels registered yet, no connection) -- used by
        ``EventWorker``'s make-before-break restart to build and connect a
        replacement before tearing down the current connection."""
        return EventHandler_SFEL(**self._source_kwargs)

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------

    def register_source(self, source_id):
        # The timing channels are always available from the main bsread header.
        if source_id in ("lab_time", "pulse_id"):
            return
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        if source_id in ("lab_time", "pulse_id"):
            return
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def get_all_source_ids(self):
        from bsread import dispatcher
        return dispatcher.get_current_channels()

    # ------------------------------------------------------------------
    # Context manager creation
    # ------------------------------------------------------------------

    def context_manager(self):
        if self.source_ids:
            from bsread import Source
            kwargs = self._source_kwargs.copy()
            kwargs["channels"] = self.source_ids
            self.source = Source(**kwargs)
        else:
            self.source = None
        return EventSourceContext(self)

    # Legacy name kept for compatibility with EventWorker.
    create_event_generator = None  # not used in new eventLoop


# ---------------------------------------------------------------------------
# Local event handler (test stream, no dispatcher)
# ---------------------------------------------------------------------------

class LocalEventHandler:
    """EventHandler that connects directly to a host:port bsread stream.

    Use this with the test stream (createStream) running on localhost.
    Bypasses the SwissFEL dispatcher entirely.

    Parameters
    ----------
    host : str
        Hostname or IP of the bsread sender (default 'localhost').
    port : int
        Port number (default 9999).
    all_channels : bool
        If True (default), receive all channels the sender broadcasts
        without explicit subscription — convenient for the test stream.
    """

    _needs_restart_on_register = False  # receives all channels; no restart needed

    def __init__(self, host="localhost", port=9999, all_channels=True):
        self.host = host
        self.port = port
        self.all_channels = all_channels
        self.source = None
        self.source_ids = []

    def clone(self):
        """See ``EventHandler_SFEL.clone``."""
        return LocalEventHandler(host=self.host, port=self.port, all_channels=self.all_channels)

    def register_source(self, source_id):
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def get_all_source_ids(self):
        """No discovery API for a direct sender; rely on live event keys instead.

        ``EventWorker._last_event_keys`` (populated from received events) is
        the accurate source of truth for a local stream and is what
        ``StreamSession.available()`` prefers; this is only the fallback.
        """
        return []

    def context_manager(self):
        from bsread import Source
        # Do NOT pass channels or all_channels — bsread would try to reconfigure
        # the source via a config socket (port+1) that the simple Sender doesn't
        # provide, causing receive() to hang.  A plain PULL connection receives
        # all channels from the data header, which is all we need.
        self.source = Source(
            host=self.host,
            port=self.port,
            receive_timeout=500,  # ms; lets stopEventLoop() exit cleanly
        )
        return EventSourceContext(self)

    create_event_generator = None


# ---------------------------------------------------------------------------
# Direct-address event handler (e.g. a cam_server pipeline's raw output stream)
# ---------------------------------------------------------------------------

class DirectStreamEventHandler:
    """EventHandler that connects directly to a raw ``tcp://host:port`` bsread
    stream via PUB/SUB, bypassing the SwissFEL dispatcher entirely.

    Distinct from :class:`LocalEventHandler` (which is tuned for the
    synthetic ``TestStream``'s PUSH/PULL sender): a cam_server pipeline's
    output is published PUB-style, matching how
    ``cam_server.PipelineClient.get_instance_message`` itself reads it
    (``bsread.source(host, port, mode=SUB)``) -- confirmed against the real
    Bernina cam_server (see ``escape/stream/example_pipeline_offload.ipynb``).

    All fields the stream publishes arrive together in every message, just
    like :class:`LocalEventHandler` -- there is no per-field subscription to
    register; a Stream just names which field to read via ``getFromSource``.

    Parameters
    ----------
    host : str
    port : int
    mode : str
        Passed straight to ``bsread.Source`` -- ``"SUB"`` (default) for a
        PUB-publishing sender such as a cam_server pipeline; ``"PULL"`` for
        a PUSH-publishing one.

    See Also
    --------
    Stream.from_tcp : the ``Stream``-level constructor built on this handler.
    """

    _needs_restart_on_register = False  # receives all fields; no restart needed

    def __init__(self, host, port, mode="SUB"):
        self.host = host
        self.port = int(port)
        self.mode = mode
        self.source = None
        self.source_ids = []

    def clone(self):
        """See ``EventHandler_SFEL.clone``."""
        return DirectStreamEventHandler(self.host, self.port, mode=self.mode)

    def register_source(self, source_id):
        if source_id not in self.source_ids:
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        try:
            self.source_ids.remove(source_id)
        except ValueError:
            pass

    def get_all_source_ids(self):
        """No discovery API for a direct stream; rely on live event keys
        instead (see LocalEventHandler.get_all_source_ids)."""
        return []

    def context_manager(self):
        from bsread import Source, SUB, PULL
        self.source = Source(
            host=self.host,
            port=self.port,
            mode=SUB if self.mode == "SUB" else PULL,
            receive_timeout=500,  # ms; lets stopEventLoop() exit cleanly
        )
        return EventSourceContext(self)

    create_event_generator = None

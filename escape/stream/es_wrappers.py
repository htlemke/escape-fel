from bsread import Source, dispatcher


class EventHandler_SFEL:
    """Example EventHandler object. The class wraps data source specific reader
    code into a standardized object with well-defined methods for
    initialisation and reading."""

    def __init__(
        self,
        source_default_keys=dict(
            host=None,
            port=9999,
            config_port=None,
            conn_type="connect",
            mode=None,
            queue_size=100,
            copy=True,
            config_address=None,
            all_channels=False,
            receive_timeout=None,
            dispatcher_url="https://dispatcher-api.psi.ch/sf",
            dispatcher_verify_request=True,
            dispatcher_disable_compression=False,
        ),
    ):
        self.source_default_keys = source_default_keys
        self.source = None
        self.source_ids = []

    def get_all_source_ids(self):
        """Dummy method which should interface to some interface providing
        availabe data sources (Detectors)."""
        return dispatcher.get_current_channels()

    def register_source(self, source_id):
        """method to register sources to be read in the loop iterator"""
        if source_id == "lab_time" or source_id == "pulse_id":
            source_id = "SIN-CVME-TIFGUN-EVR0:BUNCH-1-OK"
        if not (source_id in self.source_ids):
            self.source_ids.append(source_id)

    def remove_source(self, source_id):
        """method to remove sources from the loop iterator"""
        if source_id == "lab_time" or source_id == "pulse_id":
            return
        self.source_ids.pop(self.source_ids.index(source_id))

    def context_manager(self):
        kwargs = self.source_default_keys.copy()
        kwargs["channels"] = self.source_ids

        if not (len(kwargs["channels"]) == 0):
            self.source = Source(**kwargs)
        else:
            self.source = None
            # self.source.connect()

        return EventSource(self)

    def create_event_generator(self):
        self.source.connect()
        # return iter(EventGenerator_SFEL(self.stream))
        return iter(EventGenerator_SFEL(self.source))

    # def readStream(self,Nevents):
    # data = []
    # for n in range(Nevents):
    # m = self.s.receive()
    # data.append([m.data.data[par].value for par in pars])

    # return np.asarray(data)

    # def


class EventSource(object):
    def __init__(self, eventhandler):
        self.eventhandler = eventhandler

    def __enter__(self):
        self.eventhandler.source.connect()
        return self

    def get_event(self):
        return Event_SFEL(self.eventhandler.source.receive())

    def __exit__(self, type, value, traceback):
        self.eventhandler.source.disconnect()


class Event_SFEL:
    def __init__(self, message):
        self.message = message

    def getFromSource(self, source):
        if source == "lab_time":
            return (
                self.message.data.global_timestamp
                + 1e-9 * self.message.data.global_timestamp_offset
            )
        elif source == "pulse_id":
            return self.message.data.pulse_id

        else:
            return self.message.data.data[source].value

    def getEventId(self):
        return self.message.data.pulse_id


class EventGenerator_SFEL:
    def __init__(self, source):
        self.source = source

    def __next__(self):
        return Event_SFEL(self.source.receive())

    def __iter__(self):
        return self

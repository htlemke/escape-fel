from bsread import Source


class EventHandler_SFEL:
    """Example EventHandler object. The class wraps data source specific reader
    code into a standardized object with well-defined methods for
    initialisation and reading."""

    def __init__(self, host="localhost", port=9999, **kwargs):
        self.source = Source(host, port, **kwargs)
        self.read_Ids = []

    def getSourceIDs(self):
        """Dummy method which should interface to some interface providing
        availabe data seources (Detectors)."""
        return ["i0", "i", "t", "i_pump", "pump_on", "pulseId", "labTime"]

    def registerSource(self, sourceID):
        """Dummy method to register sources to be read in the loop iterator"""
        pass

    def eventGenerator(self):
        self.stream = self.source.connect()
        return iter(EventGenerator_SFEL(self.stream))

    # def readStream(self,Nevents):
    # data = []
    # for n in range(Nevents):
    # m = self.s.receive()
    # data.append([m.data.data[par].value for par in pars])

    # return np.asarray(data)

    # def


class Event_SFEL:
    def __init__(self, message):
        self.message = message

    def getFromSource(self, source):
        if source is "labTime":
            return (
                self.message.data.global_timestamp
                + 1e-9 * self.message.data.global_timestamp_offset
            )
        else:
            return self.message.data.data[source].value

    def getEventId(self):
        return self.message.data.pulse_id


class EventGenerator_SFEL:
    def __init__(self, stream):
        self.stream = stream

    def __next__(self):
        return Event_SFEL(self.stream.receive())

    def __iter__(self):
        return self

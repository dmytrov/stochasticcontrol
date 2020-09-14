import os
import sys
import pprint
import pickle
import numpy as np
import logging
import logging.handlers as lh


pp = pprint.PrettyPrinter(indent=4)


class LogRecord(object):
    def __init__(self, msg=None):
        self.msg = msg
        self.created = None
        self.relativecreated = None
        self.levelno = None

    def set_metadata(self, record):
        self.created = record.created
        self.relativecreated = record.relativeCreated
        self.levelno = record.levelno

    def __repr__(self):
        return pp.pformat(vars(self))


class NPRecord(LogRecord):
    def __init__(self, name, value, msg=None):
        """
        Numpy logging record

        name : string
            numpy variable name
        value : nparray
            numpy variable value
        msg : string
            a message to be logged
        """
        super(NPRecord, self).__init__(msg)
        self.name = name
        self.value = value

    def __str__(self):
        res = ""
        if self.msg is not None:
            res += "[{}]".format(self.msg)
        res += "{}".format(self.name)
        if isinstance(self.value, (int, float)):
            res += " = {}".format(str(self.value))
        elif isinstance(self.value, np.ndarray) and self.value.size < 5:
            res += " = {}".format(str(self.value))
        return res


class EventRecord(LogRecord):
    def __init__(self, record):
        super(EventRecord, self).__init__(record.msg)
        self.set_metadata(record)


class NPRecordFormatter(logging.Formatter):
    """
    Special formatter to work with NPRecord
    """
    def format(self, record):
        """
        If record is a NPRecord, formats it in a special way
        """
        if isinstance(record.msg, NPRecord):
            nprecord = record.msg
            nprecord.set_metadata(record)
            return nprecord
        else:
            return EventRecord(record)


class PickleHandler(logging.FileHandler):
    """
    Stores pickled records into a stream
    """

    def emit(self, record):
        """
        Emit a record.
        """
        msg = self.format(record)
        pickle.dump(msg, self.stream)



class NPLog(object):
    def __init__(self):
        self.records = []
        self.events = []

    def add_record(self, record):
        self.records.append(record)

    def add_event(self, event):
        self.events.append(event)

    @classmethod
    def _from_file(cls, filename):
        res = NPLog()
        with open(filename, 'rb') as f:
            try:
                while True:
                    event = pickle.load(f)
                    if isinstance(event, NPRecord):
                        res.add_record(event)
                    elif isinstance(event, EventRecord):
                        res.add_event(event)
            except EOFError:
                pass
            except:
                raise
        return res

    @classmethod
    def from_file(cls, nplogfilename):
        try:
            nplog = NPLog._from_file(nplogfilename)
        except ImportError as err:
            if err.message == "No module named copy_reg\r":
                print("Can't load the file. Trying to convert to UNIX format...")
                sbp.call(["dos2unix", nplogfilename])
                print("done.")
                nplog = NPLog._from_file(nplogfilename)
            else: 
                raise
        return nplog

    def get_names(self):
        return list(set([record.name for record in self.records]))

    def select_by_name(self, name):
        res = [record for record in self.records if record.name == name]
        if res == []:
            raise ValueError("Parameter '{}' not found".format(name))
        return res

    def stack(self, name):
        records = self.select_by_name(name)
        times = np.array([record.created for record in records])
        values = np.stack([record.value for record in records])
        return times, values


def setup_root_logger(rootlogfilename=None, removeoldlog=True, write_to_file=True):
    if rootlogfilename is None:
        rootlogfilename = "rootlog.txt"
    if write_to_file and removeoldlog:
        try:
            os.remove(rootlogfilename)
        except:
            pass

    l = logging.getLogger()
    l.setLevel(logging.DEBUG)
    while len(l.handlers) > 0:
        l.removeHandler(l.handlers[0])
    
    if write_to_file:
        f = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
        h = logging.FileHandler(rootlogfilename)
        h.setFormatter(f)
        l.addHandler(h)

    f = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(f)
    l.addHandler(h)
    return l



def setup_numpy_logger(loggername=None, nplogfilename=None, removeoldlog=True):
    if nplogfilename is None:
        nplogfilename = "numpylog.pkl"
    if removeoldlog:
        try:
            os.remove(nplogfilename)
        except:
            pass

    l = logging.getLogger(loggername)
    l.setLevel(logging.DEBUG)
    while len(l.handlers) > 0:
        l.removeHandler(l.handlers[0])
    
    f = NPRecordFormatter('[%(asctime)s][%(created)f][%(levelname)s] %(message)s')
    h = PickleHandler(nplogfilename, mode='ab')
    h.setFormatter(f)
    l.addHandler(h)
    return l




if __name__ == "__main__":
    nplogfilename = "numpylog.pkl"
    eventlogfilename = "eventlog.txt"
    try:
        os.remove(nplogfilename)
        os.remove(eventlogfilename)
    except:
        pass

    l = logging.getLogger("numpylog")
    l.setLevel(logging.DEBUG)

    f = NPRecordFormatter('[%(asctime)s][%(created)f][%(levelname)s] %(message)s')
    h = PickleHandler(nplogfilename)
    h.setFormatter(f)
    l.addHandler(h)

    f = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    h = logging.FileHandler(eventlogfilename)
    h.setFormatter(f)
    l.addHandler(h)

    f = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(f)
    l.addHandler(h)

    print("# WRITING")
    n = 10
    y = np.identity(n)
    x = np.arange(n)**2
    for i in range(n):
        l.info("Some info")
        l.debug(NPRecord("x", x[i]))
        l.debug(NPRecord("y", y[i]))
        l.debug(NPRecord("i", i))
    logging.shutdown()


    print("# READING")
    nplog = NPLog.from_file(nplogfilename)
    for event in nplog.events:
        print("[{}]{}".format(event.relativecreated, event.msg))
    print("Arrays logged: {}".format(nplog.get_names()))
    for name in nplog.get_names():
        records = nplog.select_by_name(name)
        #print("{} records: {}".format(name, records))
        print("{}: {}".format(name, nplog.stack(name)))


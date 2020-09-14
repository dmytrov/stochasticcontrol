from __future__ import print_function
import socket
import Pyro4

Pyro4.config.SOCK_NODELAY = True


class InstanceType(object):
    InProcess = 0
    Emulator = 1
    Pyro4Proxy = 2
    ChildProcPyro4Proxy = 3

    def __init__(self, mode=InProcess, objname=None, hostname=None, port=None):
        self.mode = mode
        self.objname = objname
        self.hostname = hostname
        self.port = port
        if self.objname is None:
            self.objname = "object_name"
        if self.hostname is None:
            import socket
            self.hostname = socket.gethostname()
        if self.port is None:
            self.port = 6969
        
    def URI(self):
        return "PYRO:{}@{}:{}".format(self.objname, self.hostname, self.port)


class CustomDaemon(Pyro4.Daemon):
    """
    Custom daemon to handle client disconnects
    """
    def clientDisconnect(self, conn):
        print("Client disconnected")
        shutdown_on_disconnect = False
        for obj in list(self.objectsById.values()):
            try:
                shutdown_on_disconnect = shutdown_on_disconnect or obj.shutdown_on_disconnect()
            except AttributeError:
                pass

        if shutdown_on_disconnect:
            print("Shutting down daemon...")
            self.shutdown()


def expose(obj_or_class): 
    return Pyro4.expose(obj_or_class)

def start_exposed(exposed_obj_or_class, objectId="name", port=6969):
    """
    Start the server
    """
    hostname = socket.gethostname()
    hostIP = socket.gethostbyname(hostname)
    daemon = CustomDaemon(host=hostname, port=port)
    uri = daemon.register(exposed_obj_or_class, objectId=objectId)

    print("Starting the daemon. Object URI =", uri)
    daemon.requestLoop()
    daemon.close()
    print("Daemon is closed")


def start(obj_or_class, objectId="name", port=6969):
    """
    Start the server
    """
    start_exposed(expose(obj_or_class), objectId, port)

def connect(objURI):
    print("Connecting to {}...".format(objURI))
    return Pyro4.Proxy(objURI)

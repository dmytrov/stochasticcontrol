from __future__ import print_function
import time
import Pyro4
import optotrak.ndiapiconstants

Pyro4.config.SOCK_NODELAY = True
ndi = optotrak.ndiapiconstants.NDI

# Connect to the server

optotrakURI = "PYRO:optotrak@100.1.1.2:6969"
optotrak = Pyro4.Proxy(optotrakURI)


# Init and shutdown

res = optotrak.TransputerLoadSystem("system")
print(res)
if res != 0: raise Exception(res)
time.sleep(1)

res = optotrak.TransputerInitializeSystem(ndi.OPTO_LOG_ERRORS_FLAG)
print(res)
if res != 0: raise Exception(res)

optotrak.TransputerShutdownSystem()


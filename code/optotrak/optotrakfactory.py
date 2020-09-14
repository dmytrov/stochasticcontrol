"""
Optotrak factory
"""
import optotrak.instancetype as itp
import utils.remote as rmt

def connect(instancetype=None):
    if instancetype is None:
        instancetype = itp.optotrak_default_instance_type()

    if instancetype.mode == rmt.InstanceType.InProcess:
        import pyndi as ndi
        return ndi
    elif instancetype.mode == rmt.InstanceType.Pyro4Proxy:
        return rmt.connect(instancetype.URI())
    elif instancetype.mode == rmt.InstanceType.Emulator:
        import optotrak.ndiapifunctionsemulator as oemu
        return oemu.OptoTrak
    else:
        raise NotImplementedError()

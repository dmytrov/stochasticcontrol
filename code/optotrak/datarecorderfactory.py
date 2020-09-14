"""
DataRecorder factory
"""
import time
import utils.remote as rmt
import optotrak.optotrakfactory as otf


def connect(datarecorderinstancetype=None,
            optotrakinstancetype=None):            
    if datarecorderinstancetype is None:
        datarecorderinstancetype = create_default_instance_type()
    if optotrakinstancetype is None:
        optotrakinstancetype = otf.create_default_instance_type()

    if datarecorderinstancetype.mode == rmt.InstanceType.InProcess:
        import datarecorder
        optotrak = otf.connect(optotrakinstancetype)
        return datarecorder.OptotrakDataRecorder(optotrak)
    elif datarecorderinstancetype.mode == rmt.InstanceType.Pyro4Proxy:
        return rmt.connect(datarecorderinstancetype.URI())
    elif datarecorderinstancetype.mode == rmt.InstanceType.ChildProcPyro4Proxy:
        from multiprocessing import Process
        import optotrak.datarecorderserver as drs
        proc = Process(target=drs.start, args=(datarecorderinstancetype, optotrakinstancetype))
        proc.start()
        time.sleep(0.5)
        return rmt.connect(datarecorderinstancetype.URI())
    else:
        raise NotImplementedError()


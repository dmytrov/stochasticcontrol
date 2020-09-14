import utils.remote as rmt


def optotrak_default_instance_type():
    return rmt.InstanceType(
        rmt.InstanceType.InProcess,
        objname="optotrak",
        port=6969)

def datarecorder_default_instance_type():
    return rmt.InstanceType(
        rmt.InstanceType.InProcess,
        objname="datarecorder",
        port=6970)

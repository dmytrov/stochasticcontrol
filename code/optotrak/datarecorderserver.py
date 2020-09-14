import utils.remote as rmt
import optotrak.instancetype as itp
import optotrak.datarecorder as odr
import optotrak.optotrakfactory as otf


def start(datarecorderinstancetype, optotrakinstancetype):
    optotrakobj = otf.connect(optotrakinstancetype)
    optotrakDataRecorder = rmt.expose(odr.OptotrakDataRecorder)(optotrakobj)
    rmt.start_exposed(optotrakDataRecorder,
        objectId=datarecorderinstancetype.objname,
        port=datarecorderinstancetype.port)

if __name__ == "__main__":
    optotrakit = itp.optotrak_default_instance_type()
    optotrakit.mode = rmt.InstanceType.Emulator
    start(itp.datarecorder_default_instance_type(), optotrakit)


import utils.remote as rmt
import ndiapifunctionsemulator as ndiapi
import optotrak.instancetype as itp
import optotrak.optotrakfactory as otf


def start(optotrakinstancetype):
    optotrakobj = otf.connect(optotrakinstancetype)
    rmt.start(optotrakobj,
        objectId=optotrakinstancetype.objname,
        port=optotrakinstancetype.port)

if __name__ == "__main__":
    optotrakinstancetype = itp.optotrak_default_instance_type()
    optotrakinstancetype.mode = rmt.InstanceType.Emulator
    start(optotrakinstancetype)


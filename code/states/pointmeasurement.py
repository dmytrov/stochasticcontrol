import time
import numpy as np
import psychopy.visual as pv
import statemachine.machine as sm
import statemachine.states as sms
import states.common as sc
import optotrak.calibrationcore as occ
from psychopy.visual.circle import Circle



class WaitMeasurement(sc.OptotrakReader):
    def __init__(self, timeout=1.0):
        super(WaitMeasurement, self).__init__()
        self.timeout = timeout
        self._enter_time = None
        self.set_dt(-0.2)
        self.measurement = None


    def on_enter(self):
        self._enter_time = time.time()


    def on_render(self):
        if time.time() > self._enter_time + self.timeout:
            env = self.statemachine.env
            rt = self.timed_marker_position_provider.get_marker_position(env)
            if rt is not None and not np.isnan(rt).any():
                self.measurement = rt
                self.statemachine.pop_state()



class MeasurePose(sms.Sequence):
    def __init__(self, message, timeout=2.0):
        intro = sms.TextMessage(message + ": start", timeout=2.0)
        countdown = [sms.TextMessage("Capturing in {}".format(i), timeout=1.0) for i in reversed(range(2))]
        self.waitmeasurement = WaitMeasurement()
        outtro = []
        super(MeasurePose, self).__init__([intro] + countdown + [self.waitmeasurement] + outtro)


    def get_measurement(self):
        return self.waitmeasurement.measurement



class MeasureRange(sms.Sequence):
    def __init__(self):
        intro = sms.TextMessage("Estimating motion range", timeout=2.0)
        self.idlepose = MeasurePose(message="Idle pose")
        self.flexedpose = MeasurePose(message="Flexed pose")
        self.extendedpose = MeasurePose(message="Extended pose")
        super(MeasureRange, self).__init__([intro, self.idlepose, self.flexedpose, self.extendedpose])


import time
import numpy as np
import psychopy.visual as pv
import statemachine.machine as sm
import statemachine.states as sms
import optotrak.calibrationcore as occ
from psychopy.visual.circle import Circle

class DisplayPoint(sm.State):
    def __init__(self, pointscreenposition, timeout=2.0):
        super(DisplayPoint, self).__init__()
        self.pointscreenposition = pointscreenposition
        self.timeout = timeout
        self.color = [1.0, 0.0, 0.0]
        self._enter_time = None
        self._ppcircle = None
        self.measurement = None
        
    def on_enter(self):
        self._ppcircle = pv.Circle(win=self.statemachine.env.win,
                         radius=0.015, 
                         edges=50,
                         lineColor=self.color,
                         fillColor=self.color,
                         autoDraw=False)
        self._ppcircle.setPos(self.pointscreenposition)
        self._enter_time = time.time()

    def on_render(self):
        self._ppcircle.draw()
        if time.time() > self._enter_time + self.timeout:
            env = self.statemachine.env
            t = env.last_frame_time + env.frame_period + env.monitor_delay - 0.2
            rt = env.recorder.get_value_by_timestamp(t)
            if rt is not None and not np.isnan(rt).any():
                self.measurement = rt
                self.statemachine.pop_state()


class Calibration(sms.Sequence):
    def __init__(self, calib_trans=None, timeout=2.0):
        if calib_trans is None:
            calib_trans = occ.ProjectionCalibration()
        self.render_query_points = calib_trans.render_query_points.copy()
        super(Calibration, self).__init__([DisplayPoint(point, timeout) for point in self.render_query_points])

    def get_render_query_points(self):
        return self.render_query_points

    def get_measurements(self):
        return np.vstack([state.measurement for state in self.states])

    def get_transformation(self):
        return occ.compute_optotrak_to_screen_transformation(
            self.get_render_query_points(),
            self.get_measurements())


class ProjectPoints(sm.State):
    def __init__(self, timeout=10.0):
        super(ProjectPoints, self).__init__()
        self.timeout = timeout
        self.color = [1.0, 0.0, 0.0]
        self._enter_time = None
        self._ppcircle = None
        
    def on_enter(self):
        self._ppcircle = pv.Circle(win=self.statemachine.env.win,
                         radius=0.015, 
                         edges=50,
                         lineColor=self.color,
                         fillColor=self.color,
                         autoDraw=False)
        self._enter_time = time.time()

    def on_render(self):
        env = self.statemachine.env
        t = env.last_frame_time + env.frame_period + env.monitor_delay
        rt = env.recorder.get_value_by_timestamp(t)
        if rt is not None and not np.isnan(rt).any():
            self._ppcircle.setPos(env.calib_trans.apply_calibration(rt)[0, :])
            self._ppcircle.draw()
        if time.time() > self._enter_time + self.timeout:
                self.statemachine.pop_state()

import time
import numpy as np
import psychopy.visual as pv
from psychopy.visual.circle import Circle
import utils.logger as ulog
import statemachine.machine as sm
import states.utils as su
from states.utils import DisturbanceMode


def grad_to_radians(grads):
    return grads * np.pi / 180.0



class Environment(object):
    def __init__(self):
        self.win = None
        self.recorder = None
        self.last_frame_time = None
        self.frame_period = None
        self.monitor_delay = None
        self.calib_trans = None
        self.is_debug = False
        


class ParameterIterator(sm.State):
    def __init__(self, iterstate, params):
        super(ParameterIterator, self).__init__()
        self.iterstate = iterstate  # list of states to iterate
        self.params = params
        self.index = -1
        

    def on_enter(self):
        self.index += 1
        if self.index < len(self.params):
            self.statemachine.push_state(self.interstate(self.params[i]))
        else:
            self.statemachine.pop_state()



class MarkerPositionProvider(object):
    def __init__(self, parent=None):
        super(MarkerPositionProvider, self).__init__()
        self.parent = parent

    def get_marker_position(self, env):
        raise NotImplementedError()



class TimedMarkerPositionProvider(MarkerPositionProvider):
    def __init__(self, parent=None, dt=0.0):
        assert parent is None
        super(TimedMarkerPositionProvider, self).__init__(parent)
        self.dt = dt

    def get_marker_position(self, env):
        t = env.last_frame_time + env.frame_period + env.monitor_delay + self.dt
        markerposition = env.recorder.get_value_by_timestamp(t)
        env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".markerposition", markerposition))
        return markerposition



class ScreenCalibratedMarkerPositionProvider(MarkerPositionProvider):
    def __init__(self, parent=None):
        assert parent is not None
        super(ScreenCalibratedMarkerPositionProvider, self).__init__(parent)
        
    def get_marker_position(self, env):
        markerposition = self.parent.get_marker_position(env)
        markerposition =  env.map_to_endpoint(env, markerposition)
        env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".markerposition", markerposition))
        return markerposition
        


class DisturbanceInducer(MarkerPositionProvider):
    def __init__(self, parent=None, triggercondition=None, disturbance=None):
        assert parent is not None
        super(DisturbanceInducer, self).__init__(parent)
        self.triggered = False
        self.triggercondition = triggercondition
        self.disturbance = disturbance

    def get_marker_position(self, env):
        markerposition = self.parent.get_marker_position(env)
        
        if self.triggercondition is not None and self.triggercondition(env, markerposition):
            self.triggered = True

        if self.triggered and self.disturbance is not None:
            markerposition = self.disturbance(env, markerposition)

        env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".markerposition", markerposition))
        return markerposition


class AffineDisturbanceInducer(MarkerPositionProvider):
    def __init__(self, parent=None, triggercondition=None, mtrans=None):
        #assert parent is not None
        super(AffineDisturbanceInducer, self).__init__(parent)
        if mtrans is None:
            mtrans = np.identity(3)
        self.triggered = False
        self.triggercondition = triggercondition
        self.mtrans = mtrans
        
    def get_marker_position(self, env):
        markerposition = self.parent.get_marker_position(env)
        if self.triggercondition is not None and self.triggercondition(env, markerposition):
            self.triggered = True
            env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".triggered", self.triggered))

        if self.triggered:
            markerposition = self.apply_disturbance(markerposition)

        env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".markerposition", markerposition))
        return markerposition

    def apply_disturbance(self, markerposition):
        return su.apply_disturbance(self.mtrans, markerposition)
        
    def from_translation(self, translation):
        self.mtrans = su.from_translation(translation)
        
    def from_rotation(self, angle):
        self.mtrans = su.from_rotation(angle)
        
    def from_rotation_around(self, angle, center):
        self.mtrans = su.from_rotation_around(angle, center)
        


class TimeoutState(sm.State):
    def __init__(self):
        super(TimeoutState, self).__init__()
        self._init_time = None
        self.timeouts = []  # list of tuples (timeout, handler)
        
    def add_timeout(self, timeout, handler):
        self.timeouts.append((timeout, handler))

    def on_init(self):
        super(TimeoutState, self).on_init()
        self._init_time = time.time()

    def on_render(self):
        if len(self.timeouts) > 0:
            t = time.time() - self._init_time 
            for timeout, handler in self.timeouts:
                if t > timeout:
                    handler(self)
            self.timeouts = [(timeout, handler) for timeout, handler in self.timeouts if t <= timeout]



class OptotrakReader(TimeoutState):
    def __init__(self):
        super(OptotrakReader, self).__init__()
        # Default marker position provider
        self.timed_marker_position_provider = TimedMarkerPositionProvider()
        self.calibrated_marker_position_provider = ScreenCalibratedMarkerPositionProvider(self.timed_marker_position_provider)
        self.disturbance_inducer = None
        self.get_marker_position = self.calibrated_marker_position_provider.get_marker_position    

    def set_dt(self, dt):
        self.timed_marker_position_provider.dt = dt

    def get_dt(self):
        return self.timed_marker_position_provider.dt

    def set_disturbance_inducer(self, disturbance_inducer):
        self.disturbance_inducer = disturbance_inducer
        self.disturbance_inducer.parent = self.calibrated_marker_position_provider
        self.get_marker_position = self.disturbance_inducer.get_marker_position



class WaitMarkerPosition(OptotrakReader):
    def __init__(self):
        super(WaitMarkerPosition, self).__init__()
        self.goal_position = [0, 0]
        self.goal_radius = 0.025
        self.color = [0.0, 0.0, 0.0]
        self.on_goal_reached = None  # callback event handler
        self.marker_radius = 0.015
        self.marker_visible = True
        self._ppmarker = None
        
    def get_print_info(self):
        return self.goal_position
        
    def on_init(self):
        super(WaitMarkerPosition, self).on_init()
        self._ppmarker = pv.Circle(win=self.statemachine.env.win,
                         radius=self.statemachine.env.marker_radius, 
                         edges=50,
                         lineColor=self.color,
                         fillColor=self.color,
                         autoDraw=False)
        
    def on_enter(self):
        super(WaitMarkerPosition, self).on_enter()
        env = self.statemachine.env
        env.logger.info(ulog.NPRecord("WaitMarkerPosition_params", 
            {"delay": self.get_dt(), "goal_position": self.goal_position}))

    def on_render(self):
        super(WaitMarkerPosition, self).on_render()
        env = self.statemachine.env
        rt = self.get_marker_position(env)
        if rt is not None and not np.isnan(rt).any():
            if self.marker_visible:
                env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".renderpointposition", rt))
                self._ppmarker.setPos(rt[0])
                self._ppmarker.draw()
            if np.linalg.norm(rt[0] - self.goal_position) < self.goal_radius:
                if self.on_goal_reached is not None:
                    self.on_goal_reached(self)
                self.statemachine.pop_state()



class WaitNormalizedValue(OptotrakReader):
    def __init__(self):
        super(WaitNormalizedValue, self).__init__()
        self.goal_value = [0, 0]
        self.goal_tolerance = 0.025
        self.on_goal_reached = None  # callback event handler
        self.marker_position = 0.3  # horizontal displacement w.r.t. center in screen heights
        self._pp_0 = None
        self._pp_1 = None
        self._pp_line = None
        
    def get_print_info(self):
        return self.goal_value
        
    def on_init(self):
        super(WaitNormalizedValue, self).on_init()
        self._pp_0 = pv.Circle(win=self.statemachine.env.win,
                         radius=self.statemachine.env.marker_radius, 
                         edges=50,
                         autoDraw=False)
        self._pp_1 = pv.Circle(win=self.statemachine.env.win,
                         radius=self.statemachine.env.marker_radius, 
                         edges=50,
                         autoDraw=False)
        self._pp_line = pv.Line(win=self.statemachine.env.win,
                         start=(-1.0, 0),
                         end=(1.0, 0))
        
    def on_enter(self):
        super(WaitNormalizedValue, self).on_enter()
        env = self.statemachine.env
        env.logger.info(ulog.NPRecord("WaitNormalizedValue_params", 
            {"delay": self.get_dt(), "goal_value": self.goal_value}))

    def on_render(self):
        super(WaitNormalizedValue, self).on_render()
        env = self.statemachine.env
        rt = self.calibrated_marker_position_provider.get_marker_position(env)
        if rt is not None and not np.isnan(rt).any():
            env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".renderpointposition", rt))
            self._pp_line.draw()

            x0, x1 = rt[0]
            in_range_0 = abs(x0 - self.goal_value[0]) <= self.goal_tolerance
            in_range_1 = abs(x1 - self.goal_value[1]) <= self.goal_tolerance
            in_range_color = [-1.0, 1.0, -1.0]
            out_range_color = [1.0, -1.0, -1.0]
            
            self._pp_0.setPos([-self.marker_position, 0.5 * x0])
            if in_range_0:
                self._pp_0.setFillColor(in_range_color)
                self._pp_0.setLineColor(in_range_color)
            else:
                self._pp_0.setFillColor(out_range_color)
                self._pp_0.setLineColor(out_range_color)
            self._pp_0.draw()

            self._pp_1.setPos([self.marker_position, 0.5 * x1])
            if in_range_1:
                self._pp_1.setFillColor(in_range_color)
                self._pp_1.setLineColor(in_range_color)
            else:
                self._pp_1.setFillColor(out_range_color)
                self._pp_1.setLineColor(out_range_color)
            self._pp_1.draw()

            if in_range_0 and in_range_1:
                if self.on_goal_reached is not None:
                    self.on_goal_reached(self)
                self.statemachine.pop_state()
                
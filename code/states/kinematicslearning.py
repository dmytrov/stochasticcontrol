import time
import numpy as np
import psychopy.visual as pv
import utils.logger as ulog
import statemachine.machine as sm
import statemachine.states as sms
from psychopy.visual.circle import Circle
from statemachine.states import TextMessage, Sequence
from states.pointmeasurement import MeasureRange
import utils.soundfeedback as fbk
from states.common import *



class ExperimentParams(object):
    def __init__(self):
        super(ExperimentParams, self).__init__()
        
        self.block_number = 0  # 0 for no blocks, full trial set
        self.max_blocks = 3

        self.feedback_delay = -0.1

        self.ntraining_trials = 2  # number of trials for every goal position
        self.training_timeout = 3.0
        
        self.ntest_trials = 2  # number of trials for every goal position
        self.test_timeout = 5.0
        
        self.start_position = np.array([0.0, 0.0])
        self.training_goal_positions = [[0.0, 0.4]]
        self.test_goal_positions = [[0.0, 0.4]]
        
        self.set_training_goal_positions_on_circle(grad_to_radians(np.linspace(-45, 45, num=5)))
        self.set_test_goal_positions_on_circle(grad_to_radians(np.linspace(-45, 45, num=5)))
        

    def set_training_goal_positions_on_circle(self, angles, radius=0.45):
        self.training_goal_positions = [self.start_position + radius*np.array([np.sin(angle), np.cos(angle)]) for angle in angles]


    def set_test_goal_positions_on_circle(self, angles, radius=0.45):
        self.test_goal_positions = [self.start_position + radius*np.array([np.sin(angle), np.cos(angle)]) for angle in angles]



class TrialParams(object):
    def __init__(self, number, feedback_delay, start_position, goal_position, 
                timeout=5.0, kinematics_type="xy"):
        super(TrialParams, self).__init__()
        self.number = number
        self.feedback_delay = feedback_delay
        self.start_position = start_position
        self.goal_position = goal_position
        self.timeout = timeout
        self.kinematics_type = kinematics_type
        


class XYKinematicMap(MarkerPositionProvider):
    def __init__(self, parent=None):
        #assert parent is not None
        super(XYKinematicMap, self).__init__(parent)
        self.scale = 1.0
        

    def get_marker_position(self, env):
        markerposition = self.scale * 0.5 * self.parent.get_marker_position(env)
        env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".markerposition", markerposition))
        return markerposition


class ArmKinematicMap(MarkerPositionProvider):
    def __init__(self, parent=None):
        #assert parent is not None
        super(ArmKinematicMap, self).__init__(parent)
        self.j0_rot_bias = grad_to_radians(45.0)  # w.r.t. X axis
        self.j0_translation = np.array([1.0, 0.0])  # of vertical screen
        self.j1_rot_bias = grad_to_radians(90.0)  # w.r.t. X axis
        self.j1_translation = np.array([1.0, 0.0])  # of vertical screen
        self.root_position = np.array([0.0, 0.0])
        self._set_idle_endeffector_position(np.array([0.0, 0.0]))


    def _set_idle_endeffector_position(self, pt):
        self.root_position = pt - self._get_endeffector_position(0.0, 0.0)

        
    def get_marker_position(self, env):
        angles = self.parent.get_marker_position(env)[0]  # normalized to range [-1, 1], actual data may be larger
        angles = 0.5 * np.pi * angles  # to radians
        markerposition = np.array([self._get_endeffector_position(angles[0], angles[1])])
        env.logger.info(ulog.NPRecord(self.__class__.__name__ + ".markerposition", markerposition))
        return markerposition


    def _get_endeffector_position(self, j0_angle, j1_angle):
        m = self._compute_transformation_matrix(j0_angle, j1_angle)
        return m[0:2, 2]


    def _compute_transformation_matrix(self, j0_angle, j1_angle):
        root = self._from_translation(self.root_position)
        j0_rot = self._from_rotation(self.j0_rot_bias + j0_angle)
        j0_tr = self._from_translation(self.j0_translation)
        j1_rot = self._from_rotation(self.j1_rot_bias + j1_angle)
        j1_tr = self._from_translation(self.j1_translation)
        m = root.dot(j0_rot.dot(j0_tr.dot(j1_rot.dot(j1_tr))))
        return m
        

    def _from_translation(self, translation):
        mtrans = np.identity(3)
        mtrans[0:2, 2] = translation
        return mtrans

    def _from_rotation(self, angle):
        mtrans = np.identity(3)
        mtrans[0:2, 0:2] = [[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]]
        return mtrans



class LeftArmKinematicMap(ArmKinematicMap):
    def __init__(self, parent=None):
        super(LeftArmKinematicMap, self).__init__(parent)


class RightArmKinematicMap(ArmKinematicMap):
    def __init__(self, parent=None):
        super(RightArmKinematicMap, self).__init__(parent)
        self.j0_rot_bias = grad_to_radians(45.0 + 90.0)  # w.r.t. X axis
        self.j0_translation = np.array([1.0, 0.0])  # of vertical screen
        self.j1_rot_bias = grad_to_radians(-90.0)  # w.r.t. X axis

    def _compute_transformation_matrix(self, j0_angle, j1_angle):
        root = self._from_translation(self.root_position)
        j0_rot = self._from_rotation(self.j0_rot_bias - j0_angle)
        j0_tr = self._from_translation(self.j0_translation)
        j1_rot = self._from_rotation(self.j1_rot_bias - j1_angle)
        j1_tr = self._from_translation(self.j1_translation)
        m = root.dot(j0_rot.dot(j0_tr.dot(j1_rot.dot(j1_tr))))
        return m
        


class Trial(OptotrakReader):
    def __init__(self, params=None):
        super(Trial, self).__init__()
        self.params = params
        self.start_radius = 0.020
        self.start_color = [0.0, 0.0, 0.0]
        self.goal_radius = 0.015
        self.goal_color = [0.0, 1.0, 0.0]
        self._ppstart = None
        self._ppgoal = None
        
    def get_print_info(self):
        return [self.params.number]
        
    def on_init(self):
        super(Trial, self).on_init()
        env = self.statemachine.env
        env.logger.info(ulog.NPRecord("Trial.params", self.params))
    
        marker_radius = self.statemachine.env.marker_radius
        self.start_radius = 3.0 * marker_radius
        self.goal_radius = marker_radius
        self._ppstart = pv.Circle(win=self.statemachine.env.win,
                         radius=self.start_radius, 
                         edges=50,
                         lineColor=self.start_color,
                         fillColor=self.start_color,
                         autoDraw=False)
        self._ppstart.setPos(self.params.start_position)
        self._ppgoal = pv.Circle(win=self.statemachine.env.win,
                         radius=self.goal_radius, 
                         edges=50,
                         lineColor=self.goal_color,
                         fillColor=self.goal_color,
                         autoDraw=False)
        self._ppgoal.setPos(self.params.goal_position)

        self._ppstart.setAutoDraw(False)
        self._ppgoal.setAutoDraw(False)

        if self.params.kinematics_type == "xy":
            kinematic_map = XYKinematicMap
        elif self.params.kinematics_type == "left-arm":
            kinematic_map = LeftArmKinematicMap
        elif self.params.kinematics_type == "right-arm":
            kinematic_map = RightArmKinematicMap
        else:
            raise ValueError("Unknown kinematic model '{}'".format(params.kinematics_type))

        reach_start = WaitNormalizedValue()
        reach_start.goal_tolerance = 0.05
        reach_start.on_goal_reached = self.on_start_reached

        wait_at_start = TimeoutState()
        wait_at_start.add_timeout(1.0, self.on_wait_at_start_timeout)

        reach_goal = WaitMarkerPosition()
        reach_goal.goal_position = self.params.goal_position
        reach_goal.goal_radius = self.goal_radius
        reach_goal.set_disturbance_inducer(kinematic_map())
        reach_goal.set_dt(self.params.feedback_delay)
        reach_goal.marker_visible = True
        reach_goal.color = [1.0, 1.0, 1.0]
        reach_goal.on_goal_reached = self.on_goal_reached
        reach_goal.add_timeout(0.5 * self.params.timeout, self.on_reaching_timeout_warning)
        reach_goal.add_timeout(self.params.timeout, self.on_reaching_timeout_failed)

        sequence = Sequence([reach_start, wait_at_start, reach_goal])
        sequence.on_sequence_complete = self.on_sequence_complete

        env = self.statemachine.env
        env.logger.info("Trial: sequence start")
        self.statemachine.push_state(sequence)
        

    def on_start_reached(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: start reached")
        self._ppstart.setAutoDraw(False)
        self._ppgoal.setAutoDraw(False)

    def on_wait_at_start_timeout(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: go signal")
        self._ppstart.setAutoDraw(False)
        self._ppgoal.setAutoDraw(True)
        self.statemachine.pop_state()

    def on_reaching_timeout_warning(self, sender):
        sender._ppmarker.setFillColor([1.0, 0.0, 0.0])
        sender._ppmarker.setLineColor([1.0, 0.0, 0.0])

    def on_reaching_timeout_failed(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: goal timeout")
        fbk.negative()
        self.statemachine.pop_state()
        
    def on_goal_reached(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: goal reached")
        fbk.positive()
        
    def on_sequence_complete(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: sequence end")
        self._ppgoal.setAutoDraw(False)
        self.statemachine.pop_state()



class Experiment(OptotrakReader):
    def __init__(self, params):
        super(Experiment, self).__init__()
        assert isinstance(params, ExperimentParams)
        self.params = params
    
    def on_init(self):
        super(Experiment, self).on_init()
        env = self.statemachine.env
        env.logger.info(ulog.NPRecord("Experiment.params", self.params))
        
        np.random.seed(555)

        trainingtrials = []
        for goal_position in self.params.training_goal_positions:
            for i in range(self.params.ntraining_trials):
                trialparam = TrialParams(number=0,
                                   feedback_delay=self.params.feedback_delay,
                                   start_position=self.params.start_position, 
                                   goal_position=goal_position,
                                   timeout=self.params.training_timeout,
                                   kinematics_type=self.params.kinematics_type)
                trainingtrials.append(Trial(trialparam))
        # Permute the training trials
        perms = np.random.permutation(len(trainingtrials))
        trainingtrials = [trainingtrials[i] for i in perms]

        testtrials = []
        for goal_position in self.params.test_goal_positions:
            for i in range(self.params.ntest_trials):
                trialparam = TrialParams(number=0,
                        feedback_delay=self.params.feedback_delay, 
                        start_position=self.params.start_position, 
                        goal_position=goal_position,
                        timeout=self.params.test_timeout,
                        kinematics_type=self.params.kinematics_type)
                testtrials.append(Trial(trialparam))
            
        # Permute the testtrials trials
        perms = np.random.permutation(len(testtrials))
        testtrials = [testtrials[i] for i in perms]

        print("Full number of test trial in the experiment: {}".format(len(testtrials)))

        if self.params.block_number > 0:
            ntrials = len(testtrials)
            nblocks = self.params.max_blocks
            blockid = self.params.block_number - 1
            
            min_block_size = ntrials / nblocks
            start = blockid * min_block_size
            end = start + min_block_size
            if blockid == nblocks-1:
                end = ntrials
            testtrials = [testtrials[i] for i in range(start, end)]
            print("Block {} of {}. number of disturbance trials in the block: {}".format(
                self.params.block_number, self.params.max_blocks, len(testtrials)))

            if self.params.block_number > 1:
                # Random refresh training trials for blocks > 1
                np.random.seed()
                perms = np.random.permutation(len(trainingtrials))
                trainingtrials = [trainingtrials[i] for i in perms]
                trainingtrials = [trainingtrials[i] for i in range(min(10, len(trainingtrials)))]
        else:
            print("Number of test trials: {}".format(len(testtrials)))
  
        print("Number of trainin trials: {}".format(len(trainingtrials)))
        # Set trial numbers
        trials = trainingtrials + testtrials
        for i, trial in enumerate(trials):
            trial.params.number = i

        intro = [TextMessage("Kinematics learning experiment start")]
        measurerange = MeasureRange()
        measurerange.on_sequence_complete = self.on_range_measured
        outtro = [TextMessage("Kinematics learning experiment finish")]
        sequence = Sequence([measurerange] + intro + trials + outtro)
        sequence.on_sequence_complete = self.on_sequence_complete
        env.logger.info("Experiment: sequence start")        
        self.statemachine.push_state(sequence)
        

    def on_range_measured(self, sender):
        assert isinstance(sender, MeasureRange)
        env = self.statemachine.env
        flexed = np.array(sender.flexedpose.get_measurement())
        idle = np.array(sender.idlepose.get_measurement())
        extended = np.array(sender.extendedpose.get_measurement())
        env.logger.info(ulog.NPRecord("Experiment: range flexed", flexed))
        env.logger.info(ulog.NPRecord("Experiment: range idle", idle))
        env.logger.info(ulog.NPRecord("Experiment: range extended", extended))
        env.calib_trans.compute_calibration_range(flexed, idle, extended)
        

    def on_sequence_complete(self, sender):
        env = self.statemachine.env
        env.logger.info("Experiment: sequence end")
        self.statemachine.pop_state()


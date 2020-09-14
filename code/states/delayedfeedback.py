import time
import numpy as np
import psychopy.visual as pv
import utils.logger as ulog
import statemachine.machine as sm
import statemachine.states as sms
from psychopy.visual.circle import Circle
from statemachine.states import TextMessage, Sequence
import utils.soundfeedback as fbk
from states.common import *




class DigitalStream(object):
    TRIAL_BIT = 0x01


class CoordSystem(object):
    Screen = 0
    Task = 1



class ExperimentParams(object):
    def __init__(self):
        super(ExperimentParams, self).__init__()
        
        self.block_number = 0  # 0 for no blocks, full trial set
        self.max_blocks = 3

        self.feedback_delay = -0.1

        self.ntraining_trials = 2  # number of trials for every goal position
        self.training_timeout = 0.75
        self.disturbance_timeout = 5.0
        
        self.start_position = np.array([0.0, -0.45])
        self.goal_positions = [[0.0, 0.0]]
        
        self.nrotation_disturbance_trials = 5  # number of trials for every disturbance
        self.rotation_disturbances = grad_to_radians(np.linspace(-30.0, 30.0, num=5))  # degrees
        
        self.ntranslation_disturbance_trials = 5  # number of trials for every disturbance
        self.translation_disturbances = [[0.0, 0.0]]  # screen height units
        
        self.disturbance_CS = CoordSystem.Task
        self.disturbance_threshold = 2.0  # onset threshold, ratio of goal/start distances

        self.set_goal_positions_on_circle(grad_to_radians(np.linspace(-45, 45, num=5)))
        self.set_translation_x_disturbances(np.linspace(-0.1, 0.1, num=5))

        self.digital_io_device_number = 0
        

    def set_goal_positions_on_circle(self, angles, radius=0.65):
        self.goal_positions = [self.start_position + radius*np.array([np.sin(angle), np.cos(angle)]) for angle in angles]


    def set_translation_x_disturbances(self, x_disturbances):
        self.translation_disturbances = [[x_disturbance, 0.0] for x_disturbance in x_disturbances]



class TrialParams(object):
    def __init__(self, number, feedback_delay, start_position, goal_position, 
                disturbance_threshold=None,
                disturbance_mode=DisturbanceMode.Rotation,
                disturbance_value=0.0,
                timeout=5.0):
        super(TrialParams, self).__init__()
        self.number = number
        self.feedback_delay = feedback_delay
        self.start_position = start_position
        self.goal_position = goal_position
        self.disturbance_threshold = disturbance_threshold
        self.disturbance_mode = disturbance_mode
        self.disturbance_value = disturbance_value
        self.timeout = timeout
        


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
        return [self.params.number, self.params.disturbance_threshold]
        
    def on_init(self):
        super(Trial, self).on_init()
        env = self.statemachine.env
        env.logger.info(ulog.NPRecord("Trial.params", self.params))
    
        marker_radius = self.statemachine.env.marker_radius
        self.start_radius = 2.0 * marker_radius
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

        self._ppstart.setAutoDraw(True)
        self._ppgoal.setAutoDraw(False)

        reach_start = WaitMarkerPosition()
        reach_start.goal_position = self.params.start_position
        reach_start.goal_radius = self.start_radius
        reach_start.marker_visible = False
        reach_start.on_goal_reached = self.on_start_reached

        wait_at_start = TimeoutState()
        wait_at_start.add_timeout(1.0, self.on_wait_at_start_timeout)

        disturbance_inducer = AffineDisturbanceInducer()
        if self.params.disturbance_mode == DisturbanceMode.Translation:
            disturbance_inducer.from_translation(self.params.disturbance_value)
        elif self.params.disturbance_mode == DisturbanceMode.Rotation:
            disturbance_inducer.from_rotation_around( \
                self.params.disturbance_value,
                self.params.start_position)
        disturbance_inducer.triggercondition = self.disturbance_triggercondition
        
        reach_goal = WaitMarkerPosition()
        reach_goal.goal_position = self.params.goal_position
        reach_goal.goal_radius = self.goal_radius
        reach_goal.set_dt(self.params.feedback_delay)
        reach_goal.marker_visible = True
        reach_goal.color = [1.0, 1.0, 1.0]
        reach_goal.on_goal_reached = self.on_goal_reached
        reach_goal.set_disturbance_inducer(disturbance_inducer)
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
        env.digital_io.set_bit(DigitalStream.TRIAL_BIT)
        self._ppstart.setAutoDraw(False)
        self._ppgoal.setAutoDraw(True)
        self.statemachine.pop_state()

    def on_reaching_timeout_warning(self, sender):
        sender._ppmarker.setFillColor([1.0, 0.0, 0.0])
        sender._ppmarker.setLineColor([1.0, 0.0, 0.0])

    def on_reaching_timeout_failed(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: goal timeout")
        env.digital_io.reset_bit(DigitalStream.TRIAL_BIT)
        fbk.negative()
        self.statemachine.pop_state()
        
    def on_goal_reached(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: goal reached")
        env.digital_io.reset_bit(DigitalStream.TRIAL_BIT)
        fbk.positive()
        
    def on_sequence_complete(self, sender):
        env = self.statemachine.env
        env.logger.info("Trial: sequence end")
        self._ppgoal.setAutoDraw(False)
        self.statemachine.pop_state()

    def disturbance_triggercondition(self, env, markerposition):
        if self.params.disturbance_threshold is None:
            return False
        disttostart = np.linalg.norm(markerposition - self.params.start_position)
        disttogoal = np.linalg.norm(markerposition - self.params.goal_position)
        return disttogoal < self.params.disturbance_threshold * disttostart



class Experiment(OptotrakReader):
    def __init__(self, params):
        super(Experiment, self).__init__()
        assert isinstance(params, ExperimentParams)
        self.params = params
        
    
    def on_init(self):
        super(Experiment, self).on_init()
        env = self.statemachine.env
        env.logger.info(ulog.NPRecord("Experiment.params", self.params))
        env.digital_io.reset_bit(DigitalStream.TRIAL_BIT)        
        np.random.seed(555)

        trainingtrials = []
        for goal_position in self.params.goal_positions:
            for i in range(self.params.ntraining_trials):
                trialparam = TrialParams(number=0,
                                   feedback_delay=self.params.feedback_delay,
                                   start_position=self.params.start_position, 
                                   goal_position=goal_position,
                                   disturbance_mode=DisturbanceMode.NoDisturbance,
                                   disturbance_threshold=None,
                                   timeout=self.params.training_timeout)
                trainingtrials.append(Trial(trialparam))
        # Permute the training trials
        perms = np.random.permutation(len(trainingtrials))
        trainingtrials = [trainingtrials[i] for i in perms]

        catchtrials = []
        for goal_position in self.params.goal_positions:
            for disturbance in self.params.rotation_disturbances:
                for i in range(self.params.nrotation_disturbance_trials):
                    trialparam = TrialParams(number=0,
                            feedback_delay=self.params.feedback_delay, 
                            start_position=self.params.start_position, 
                            goal_position=goal_position,
                            disturbance_threshold=self.params.disturbance_threshold,
                            disturbance_mode=DisturbanceMode.Rotation,
                            disturbance_value=disturbance,
                            timeout=self.params.disturbance_timeout)
                    catchtrials.append(Trial(trialparam))
            for disturbance in self.params.translation_disturbances:
                for i in range(self.params.ntranslation_disturbance_trials):
                    if self.params.disturbance_CS == CoordSystem.Screen:
                        disturbance_value = disturbance
                    elif self.params.disturbance_CS == CoordSystem.Task:
                        v = goal_position - self.params.start_position
                        angle = np.arctan2(v[1], v[0]) - 0.5 * np.pi
                        mr = np.array([[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]])
                        disturbance_value = mr.dot(disturbance)
                    trialparam = TrialParams(number=0,
                            feedback_delay=self.params.feedback_delay, 
                            start_position=self.params.start_position, 
                            goal_position=goal_position,
                            disturbance_threshold=self.params.disturbance_threshold,
                            disturbance_mode=DisturbanceMode.Translation,
                            disturbance_value=disturbance_value,
                            timeout=self.params.disturbance_timeout)
                    catchtrials.append(Trial(trialparam))

        # Permute the catch trials
        perms = np.random.permutation(len(catchtrials))
        catchtrials = [catchtrials[i] for i in perms]

        print("Full number of disturbance trials in the experiment: {}".format(len(catchtrials)))

        if self.params.max_blocks > 1 and self.params.block_number != 0:
            ntrials = len(catchtrials)
            nblocks = self.params.max_blocks
            blockid = self.params.block_number - 1
            
            min_block_size = ntrials / nblocks
            start = blockid * min_block_size
            end = start + min_block_size
            if blockid == nblocks-1:
                end = ntrials
            catchtrials = [catchtrials[i] for i in range(start, end)]
            print("Block {} of {}. number of disturbance trials in the block: {}".format(
                self.params.block_number, self.params.max_blocks, len(catchtrials)))
            
            if self.params.block_number > 1:
                # Random refresh training trials for blocks > 1
                np.random.seed()
                perms = np.random.permutation(len(trainingtrials))
                trainingtrials = [trainingtrials[i] for i in perms]
                trainingtrials = [trainingtrials[i] for i in range(min(10, len(trainingtrials)))]
        
        print("Number of trainin trials: {}".format(len(trainingtrials)))
        print("Number of disturbance trials: {}".format(len(catchtrials)))
  
        # Set trial numbers
        trials = trainingtrials + catchtrials
        for i, trial in enumerate(trials):
            trial.params.number = i

        trailssequence = Sequence(trials)
        trailssequence.on_step = self.on_trial_step

        intro = [TextMessage("Delayed feedback experiment start")]
        outtro = [TextMessage("Delayed feedback experiment finish")]
        sequence = Sequence(intro + [trailssequence] + outtro)
        sequence.on_sequence_complete = self.on_sequence_complete
        env.logger.info("Experiment: sequence start")        
        self.statemachine.push_state(sequence)
        
    def on_trial_step(self, sender):
        i = sender.index + 1
        n = len(sender.states)
        print("Trial {} of {} ({}%)".format(i, n, 100.0 * i / n))

    def on_sequence_complete(self, sender):
        env = self.statemachine.env
        env.logger.info("Experiment: sequence end")
        self.statemachine.pop_state()


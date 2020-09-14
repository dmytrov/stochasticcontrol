import sys
import os
import gc
from multiprocessing import freeze_support
import time
import platform
import traceback
import argparse
import numpy as np
from psychopy import visual, core, event #import some libraries from PsychoPy
from utils.realtimeclock import rtc
from utils.realtimeclock import highrestimer
from optotrak.datarecorder import OptotrakDataRecorder
import optotrak.instancetype as itp
import optotrak.datarecorderfactory as drf
import utils.remote as rmt
import matplotlib.pyplot as plt
import optotrak.ndiapiconstants
from optotrak.ndiapiconstants import NDI as ndi
import optotrak.calibrationcore as cc
import statemachine.machine as sm
from statemachine.states import TextMessage, Sequence
from states.common import grad_to_radians
from states.calibration import Calibration, ProjectPoints
from states.kinematicslearning import WaitMarkerPosition, Trial, Experiment, ExperimentParams
import experiments.calibratealignment as cal
import states.common as sc
import utils.logger as ulog
import experiments.common as ec


using_windows_os = platform.system() == "Windows"
is_debug = not using_windows_os

 
class Environment(sc.Environment):
    def __init__(self):
        super(Environment, self).__init__()
        self.monitor_delay = None
        self.marker_radius = None
        self.logger = None
        self.map_to_endpoint = None
        self.kinematics_type = 'xy' 
        

def main(params):
    mywin = None
    recorder = None
    try:
        savedirectory = time.strftime("../../recordings/kinematicslearning/%Y-%m-%d-%H.%M.%S")
        if not os.path.exists(savedirectory):
            os.makedirs(savedirectory)

        verbose = False
        fullscreen = not is_debug
        units = "height"
        if fullscreen:
            mywin = visual.Window(
                size=(1920, 1080),
                fullscr=True,
                screen=1,
                allowGUI=False,
                allowStencil=False,
                winType='pyglet',  # 'pyglet', 'pygame'
                monitor=u'testMonitor',
                color=[-1, -1, -1],
                colorSpace='rgb',
                blendMode='avg',
                useFBO=True,
                units=units,
                viewScale=[-1, 1],  # Correct for the mirrored screen
                waitBlanking=False)
        else:
            mywin = visual.Window(
                size=[800, 600], 
                monitor="testMonitor", 
                color=[-1, -1, -1], 
                units=units)
        
        datafps = 120.0
        features_func = ec.map_to_angles_from_mouse_emulation if is_debug else ec.map_to_angles
        env = Environment()
        #env.logger = ulog.setup_root_logger(rootlogfilename=savedirectory+"/kinematicslearning.txt")
        env.logger = ulog.setup_numpy_logger(nplogfilename=savedirectory+"/kinematicslearning.pkl")
        env.win = mywin
        env.frame_period = 1.0/datafps
        env.monitor_delay = 1.0/120.0
        env.marker_radius = 0.015
        env.calib_trans = cc.NormalizedRangeCalibration(features_func=features_func)
        env.map_to_endpoint = ec.map_calibrated_range
        env.is_debug = is_debug
        statemachine = sm.StateMachine(env)
            
        nmarkers = 8
        datarecorderinstancetype = itp.datarecorder_default_instance_type()
        datarecorderinstancetype.mode = rmt.InstanceType.InProcess if is_debug else rmt.InstanceType.ChildProcPyro4Proxy
        optotrakinstancetype = itp.optotrak_default_instance_type()
        optotrakinstancetype.mode = rmt.InstanceType.Emulator if is_debug else rmt.InstanceType.InProcess
        recorder = drf.connect(datarecorderinstancetype, optotrakinstancetype)
        if is_debug: recorder.optotrak.getmousecoordsfunc = event.Mouse().getPos
        recorder.sleep_period = 0
        recorder.verbose = False
        recorder.init_optotrak(nplogfilename=savedirectory+"/datarecorder.pkl")
        recorder.setup_odau_collection(numchannels=4, collecttime=3600.0, collection_frequency=1000.0,
            datafilename=savedirectory+"/REC-001.ODAU")
        recorder.setup_optotrak_collection(nummarkers=nmarkers, collecttime=3600.0,
            datafps=datafps, buffer_capacity=datafps, 
            datafilename=savedirectory+"/REC-001.OPTO",)
        recorder.start_pulling_thread()
        recorder.waitfillbuffer()
        env.recorder = recorder
        #cal.calibrate(env, statemachine)
        
        statemachine.push_state(Experiment(params))
        env.last_frame_time = rtc()
        while statemachine.get_current_state() is not None:
            statemachine.render()
            mywin.flip(clearBuffer=True)
            env.last_frame_time = rtc()
            keys = event.getKeys()
            if len(keys) > 0 and keys[0] == "q":
                break

    except Exception as e:
            print(traceback.format_exc())
            raise
    finally:
        if recorder is not None:
            recorder.stop_pulling_thread()
            if using_windows_os:
                import optotrak.filereader as ofr
                ofr.convert_all_files_in_filesystem_subtree(savedirectory)
        if mywin is not None:
            mywin.close()
        core.quit()


if __name__ == "__main__":
    freeze_support()
    gc.disable()

    parser = argparse.ArgumentParser(
        prog=__file__,
        description="Virtual arm kinematics learning exeriment",
        epilog="""\
        Unauthorized copying and/or use of this software is strictly prohibited.
        This software comes with no warranty.
        Written by Dmytro Velychko <velychko@staff.uni-marburg.de>
        University of Marburg, Department of Psychology
        AE Endres, Theoretical Neuroscience Lab""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--v", action='version', version='%(prog)s 0.1')
    parser.add_argument("--b", type=int, default=0, help="Run only B-th block, 0 for all blocks. Default is 0")
    parser.add_argument("--n", type=int, default=3, help="Full number of blocks. Default is 3")
    parser.add_argument("--t", type=float, default=-0.1, help="Feedback time shift, range [-1.0, 1.0] sec. Default is -0.1")
    parser.add_argument("--k", default="xy", help="Kinematic model {'xy', 'left-arm', 'right-arm'}. Default is 'xy'")
    args = parser.parse_args()
    
    params = ExperimentParams()
    params.feedback_delay = args.t
    params.block_number = args.b
    params.max_blocks = args.n
    params.kinematics_type = args.k
    params.ntraining_trials = 2  # number of trials for every goal position
    params.ntest_trials = 2
    params.start_position = np.array([0.0, 0.0])
    params.training_timeout = 50.0
    params.test_timeout = 50.0
    params.set_training_goal_positions_on_circle(grad_to_radians(np.linspace(-45, 45, num=5)))
    params.set_test_goal_positions_on_circle(grad_to_radians(180 + np.linspace(-90, 90, num=5)))
    if params.block_number > params.max_blocks:
        raise ValueError("Block number is larger than max number of blocks")
    if params.kinematics_type not in set(("xy", "left-arm", "right-arm")):
        raise ValueError("Unknown kinematic model '{}'".format(params.kinematics_type))

    if using_windows_os:
        import wres
        with wres.set_resolution():  # set max windows timer resoltion, 1 msec
            main(params)
    else:
        main(params)
    



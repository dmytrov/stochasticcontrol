import sys
import gc
from multiprocessing import freeze_support
import time
import platform
using_windows_os = platform.system() == "Windows"
if using_windows_os:
    import wres
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
from states.calibration import Calibration, ProjectPoints
import states.common as sc



def calibrate(env, statemachine, force_recalibrate=False):
    if force_recalibrate or not env.calib_trans.load_from_file():
        timeout = 2.5
        calibrationstate = Calibration(env.calib_trans, timeout=timeout)
        sequence = Sequence([TextMessage("Calibration start"), 
                             calibrationstate,
                             TextMessage("Calibration finish")])
        nstatesstart = len(statemachine.statestack)
        statemachine.push_state(sequence)
        env.last_frame_time = rtc()
        while nstatesstart < len(statemachine.statestack):
            statemachine.render()
            env.win.flip(clearBuffer=True)
            env.last_frame_time = rtc()
        env.calib_trans.M_calibration = calibrationstate.get_transformation()
        print(env.calib_trans.M_calibration)
        env.calib_trans.save_to_file()
    env.calib_trans.load_from_file()


def main():
    debug = not using_windows_os
    verbose = False
    fullscreen = not debug
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
            waitBlanking=False)
    else:
        mywin = visual.Window(
            size=[800, 600], 
            monitor="testMonitor", 
            color=[-1, -1, -1], 
            units=units)
        mywin.setMouseVisible(True)
    
    datafps = 120.0
    env = sc.Environment()
    env.win = mywin
    env.frame_period = 1.0/datafps
    env.monitor_delay = 1.0/120.0
    env.calib_trans = cc.ProjectionCalibration(mywin)
    statemachine = sm.StateMachine(env)

    nmarkers = 1
    datarecorderinstancetype = itp.datarecorder_default_instance_type()
    datarecorderinstancetype.mode = rmt.InstanceType.InProcess if debug else rmt.InstanceType.ChildProcPyro4Proxy
    optotrakinstancetype = itp.optotrak_default_instance_type()
    optotrakinstancetype.mode = rmt.InstanceType.Emulator if debug else rmt.InstanceType.InProcess
    recorder = drf.connect(datarecorderinstancetype, optotrakinstancetype)
    if debug: recorder.optotrak.getmousecoordsfunc = event.Mouse().getPos
    recorder.sleep_period = 0
    recorder.verbose = False
    recorder.init_optotrak(nummarkers=nmarkers, collecttime=100.0, datafps=datafps, buffer_capacity=datafps)
    recorder.start_pulling_thread()
    recorder.waitfillbuffer()

    env.recorder = recorder
    
    calibrate(env, statemachine)
    
    statemachine.push_state(ProjectPoints())
    env.last_frame_time = rtc()
    while statemachine.get_current_state() is not None:
        statemachine.render()
        mywin.flip(clearBuffer=True)
        env.last_frame_time = rtc()

    recorder.stop_pulling_thread()
    mywin.close()
    core.quit()


if __name__ == "__main__":
    freeze_support()
    gc.disable()
    if using_windows_os:
        with wres.set_resolution():  # set max windows timer resoltion, 1 msec
            main()
    else:
        main()
    



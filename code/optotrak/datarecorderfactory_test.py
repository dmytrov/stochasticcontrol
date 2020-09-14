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
ndi = optotrak.ndiapiconstants.NDI

def get_frame_data(nmarkers, frameNumber, framerate):
        bias = np.array([0.0, 0.0, -2500.0])
        factor = 1.0 * np.array([1.0, -1.0, 1.0])
        freq = 0.3
        phases = np.array([freq * np.pi * frameNumber / framerate, 
                           freq * np.pi * frameNumber / framerate + 0.5 * np.pi, 
                           freq * 1.33 * np.pi * frameNumber / framerate])
        sines = 100.0 * np.vstack([np.sin(phases + np.pi * i/nmarkers) for i in range(nmarkers)])
        data = factor * sines + bias
        return data

def main():
    # pygame seems to be faster than pyglet
    # pyglet is the default library
    verbose = False
    fullscreen = True
    if fullscreen:
        mywin = visual.Window(size=(1920, 1080), fullscr=True, screen=1, allowGUI=False, allowStencil=False,
            winType='pyglet',  # 'pyglet', 'pygame'
            monitor=u'testMonitor',
            color=[-1, -1, -1],
            colorSpace='rgb',
            blendMode='avg',
            useFBO=True,
            units="pix",
            waitBlanking=False
            )
    else:
        mywin = visual.Window([1920, 1080], monitor="testMonitor", units="deg")

    # Create some stimuli
    nmarkers = 8
    datafps = 120
        
    #fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, rgb=-1)
    circles = [visual.Circle(win=mywin,
                            radius=0.5, 
                            edges=100,
                            lineColor=[1.0, 0.0, 0.0],
                            fillColor=[1.0, 0.0, 0.0],
                            autoDraw=False)
                for i in range(nmarkers)]
    for i in range(nmarkers):
        circles[i].setPos([0.0, 0.0])

    
    datarecorderinstancetype = itp.datarecorder_default_instance_type()
    datarecorderinstancetype.mode = rmt.InstanceType.ChildProcPyro4Proxy
    optotrakinstancetype = itp.optotrak_default_instance_type()
    optotrakinstancetype.mode = rmt.InstanceType.Emulator

    recorder = drf.connect(datarecorderinstancetype, optotrakinstancetype)
    recorder.sleep_period = 0
    recorder.verbose = False
    recorder.init_optotrak(nummarkers=nmarkers, 
        collecttime=160.0, datafps=datafps, buffer_capacity=200)
    recorder.start_pulling_thread()
    recorder.waitfillbuffer()

    bias = np.array([0.0, 0.0, -2500.0])
    factor = 0.1 * np.array([1.0, -1.0, 1.0])

    t0 = time.time()
    framecount = 0
    lastreceivedframenumber = 0
    framestowait = 100
    lf = time.time()
    last_frame_time = rtc()
    monitor_delay = 1.0/datafps
    frame_period = 1.0/datafps
    experiment_delay = -0.5
    frametimes = []
    
    rt = 0
    tstart = rtc()
    n = 0
    while rt is not None:
        #time.sleep(5)
        t = last_frame_time + frame_period + monitor_delay + experiment_delay
        
        #rt = recorder.get_value_by_timestamp(t)
        
        x, y = recorder.select_values_by_timestamp(t)
        if x is not None:
            rt = OptotrakDataRecorder.GP_prediction(x, y, t)
        else:
            rt = None
        
        n += 1
        #if n > 600: n = 0
        #rt = get_frame_data(nmarkers, 2*n, 60.0)
        
        if rt is not None: 
            x = factor * (rt - bias)
            for i in range(2):
                circles[i].setPos(x[i, 1:3])
                circles[i].draw()
            mywin.flip(clearBuffer=True)
            last_frame_time = rtc()
            frametimes.append(last_frame_time)

            if verbose:
                framecount += 1
                if framecount >= framestowait:
                    t1 = time.time()
                    if t1-t0 > 1e-6:
                        print("Timed frame frequency: {}".format(framestowait / (t1-t0)))
                    framecount = 0
                    t0 = t1
        if len(event.getKeys())>0:
            recorder.stop_pulling_thread()
            rt = None
        event.clearEvents()

    mywin.close()
    
    screenfps = 60.0
    frametimes = np.array(frametimes)
    frameintervals = np.diff(frametimes)
    plt.plot(1.0/screenfps + 0*frametimes, color="green")
    plt.plot(2.0/screenfps + 0*frametimes, color="red")
    plt.plot(frameintervals)
    plt.show()

    core.quit()

if __name__ == "__main__":
    freeze_support()
    gc.disable()
    if using_windows_os:
        with wres.set_resolution():  # set max windows timer resoltion, 1 msec
            main()
    else:
        main()
    



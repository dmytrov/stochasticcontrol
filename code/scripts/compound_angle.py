import sys
import time
import numpy as np
from psychopy import visual, core, event #import some libraries from PsychoPy
import optotrak.ndiapiconstants
import optotrak.client as oclient
import optotrak.datarecorder as dr
import linalg.routines as lr

ndi = optotrak.ndiapiconstants.NDI
opt = oclient.connect(oclient.InstanceType.Emulator)

# pygame seems to be faster than pyglet
# pyglet is the default library
windowed = True
if windowed:
    mywin = visual.Window([800,600], monitor="testMonitor", units="deg") 
else:
    mywin = visual.Window(size=(1920, 1080), fullscr=True, screen=0, allowGUI=False, allowStencil=False,
        winType='pyglet',  # 'pyglet', 'pygame'
        monitor=u'testMonitor',
        color=[0, 0, 0],
        colorSpace='rgb',
        blendMode='avg',
        useFBO=False,
        units="cm",
        #waitBlanking=True
        )

#create some stimuli
nmarkers = 4
    
fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0,0], sf=0, rgb=-1)
circles = [visual.Circle(win=mywin,
                         radius=0.5, 
                         edges=100,
                         lineColor=[1.0, 0.0, 0.0],
                         fillColor=[1.0, 0.0, 0.0],
                         autoDraw=False)
            for i in range(nmarkers)]
for i in range(nmarkers):
    circles[i].setPos([0.0, 0.0])

arm_endpoint = visual.Circle(win=mywin,
                         radius=0.5, 
                         edges=100,
                         lineColor=[1.0, 0.0, 0.0],
                         fillColor=[1.0, 0.0, 0.0],
                         autoDraw=False)

if True:
    recorder = dr.OptotrakDataRecorder(opt)
    recorder.sleep_period = 0.001
    recorder.init_optotrak(nummarkers=nmarkers, collecttime=10.0, datafps=120)
    recorder.start_pulling_thread()
    while recorder.realtimedatabuffer.is_empty():
        time.sleep(0.1)

    bias = np.array([0.0, 0.0, -2500.0])
    factor = 0.1 * np.array([1.0, -1.0, 1.0])

    uSpoolComplete = False
    t0 = time.time()
    framecount = 0
    lastreceivedframenumber = 0
    framestowait = 100
    lf = time.time()
    
    while recorder.pullingthread.isAlive():
        rt = recorder.get_value_by_timestamp(-0.1)
        x = factor * (rt - bias)
        angle = lr.lines_angle(x[0], x[1], x[2], x[3])
        arm_endpoint.setPos([angle, 0])
        arm_endpoint.draw()
        mywin.flip(clearBuffer=True)

        framecount += 1
        if framecount >= framestowait:
            t1 = time.time()
            if t1-t0 > 1e-6:
                print("Timed frame frequency: {}".format(framestowait / (t1-t0)))
            framecount = 0
            t0 = t1
        res = opt.RequestLatest3D()
        if res != 0: raise Exception(res)

        if len(event.getKeys())>0:
            recorder.stop_pulling_thread()
        event.clearEvents()


    mywin.close()
    core.quit()
    



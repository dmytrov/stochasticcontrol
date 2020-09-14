import sys
import time
import numpy as np
from psychopy import visual, core, event #import some libraries from PsychoPy
import optotrak.ndiapiconstants
import optotrak.client as oclient
import optotrak.datapuller as dp


ndi = optotrak.ndiapiconstants.NDI
opt = oclient.connect(oclient.InstanceType.Pyro4Proxy)


# pygame seems to be faster than pyglet
# pyglet is the default library
fullscreen = False
if fullscreen:
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
else:
    mywin = visual.Window([800,600], monitor="testMonitor", units="deg")

#create some stimuli
nmarkers = 2
    
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

if True:
    datapuller = dp.OptotrakDataPuller(opt)
    datapuller.sleep_period = 0.001
    datapuller.print_messages = False
    datapuller.init_optotrak(nummarkers=nmarkers, collecttime=100.0, datafps=120.0)
    datapuller.start_pulling_thread()
    while datapuller.realtimedatabuffer.is_empty() and datapuller.pullingthread.isAlive():
        time.sleep(0.1)

    bias = np.array([0.0, 0.0, -2500.0])
    factor = 0.1 * np.array([1.0, -1.0, 1.0])

    uSpoolComplete = False
    t0 = time.time()
    framecount = 0
    lastreceivedframenumber = 0
    framestowait = 100
    lf = time.time()
    
    while datapuller.pullingthread.isAlive():
        rt = datapuller.realtimedatabuffer.get_latest()
        #print(rt.framenr)
        if lastreceivedframenumber < rt.framenr:
            if rt.framenr - lastreceivedframenumber > 1:
                print("FRAMES SKIPPED: {}".format(rt.framenr - lastreceivedframenumber))
            lastreceivedframenumber = rt.framenr
            x = factor * (np.array(rt.data) - bias)
            for i in range(nmarkers):
                circles[i].setPos(x[i, 1:3])
                circles[i].draw()
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
                datapuller.stop_pulling_thread()
            event.clearEvents()


    mywin.close()
    core.quit()
    



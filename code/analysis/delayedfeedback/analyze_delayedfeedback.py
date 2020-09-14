import os
import sys
import subprocess as sbp
import pprint
from utils.logger import *
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)



def get_nprecord_channels(nplog):
    pass



class Trial(object):
    def __init__(self, record):
        self._record = record
        self.params = record.value
        self.t_start = record.created
        self.t_go = None
        self.t_end = None
        self.events = []
        self.opto = None
        self.odau = None
        

    def add_event(self, event):
        self.events.append(event)
        if event.msg == "Trial: go signal":
            self.t_go = event.created

    
    def select_optotrak_data(self, opto, odau, trecordingstart, trecordingend):
        nopto = len(opto)
        nodau = len(odau)
        i0 = (self.t_go - trecordingstart) / (trecordingend - trecordingstart)
        i1 = (self.t_end - trecordingstart) / (trecordingend - trecordingstart)
        self.opto = opto[int(nopto * i0) : int(nopto * i1)]
        self.odau = opto[int(nodau * i0) : int(nodau * i1)]



class Block(object):
    def __init__(self, nplog):
        self.nplog = nplog
        self.params = nplog.select_by_name("Experiment.params")[0]
        self.trials = []
        self.add_trial_records(nplog.select_by_name("Trial.params"))
        self.sort_events(nplog.events)
        

    def add_trial_records(self, recs):
        self.trials.extend([Trial(rec) for rec in recs])
        

    def sort_events(self, events):
        for event in events:
            print(event.msg)
            if event.msg == "Trial: sequence end":
                for trial in reversed(self.trials):
                    if event.created > trial.t_start:
                        trial.t_end = event.created
                        break

        for i in range(len(self.trials)-1):
            self.trials[i].t_end = self.trials[i+1].t_start

        for event in events:
            for trial in self.trials:
                if event.created >= trial.t_start and event.created < trial.t_end:
                    trial.add_event(event)
                    break


    def _slect_by_time(self, t, x, t0, t1):
        ind = ((t >= t0) & (t < t1))
        return t[ind], x[ind]


    def plot_trials(self, prefix=None):
        if prefix is None:
            prefix = "block-x-"
        rpt, rp = self.nplog.stack("WaitMarkerPosition.renderpointposition")
        sct, sc = self.nplog.stack("ScreenCalibratedMarkerPositionProvider.markerposition")
        rp = np.squeeze(rp)
        sc = np.squeeze(sc)
        print(rp.shape, sc.shape)
        plots_dir = "./plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        for i, trial in enumerate(self.trials[:]):
            plt.figure(figsize=(5, 3))
            t, x = self._slect_by_time(rpt, rp, trial.t_go, trial.t_end)
            print(x.shape)
            plt.plot(x[:, 0], x[:, 1], ".", alpha=0.2)
            t, x = self._slect_by_time(sct, sc, trial.t_go, trial.t_end)
            plt.plot(x[:, 0], x[:, 1], ".", alpha=0.2)
            plt.xlim([-0.7, 0.7])
            plt.ylim([-0.5, 0.3])
            #plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(plots_dir, "{}{}.pdf".format(prefix, i)))

            
            


if __name__ == "__main__":
    subjectpathes = ["../../../data/delayedfeedback/2017-06-12-(Gunnar)",
                     "../../../data/delayedfeedback/2017-08-12-(Felix-1)"]
    subjectpath = subjectpathes[1] 
    print("Session path \"{}\"".format(subjectpath))
    blockpaths = [x[0] for x in os.walk(subjectpath)][1:]
    print("Found blocks:")
    for blockpath in blockpaths:
        print(blockpath)
    
    iblock = 0
    blockpath = blockpaths[iblock]
    print("Reading block {}".format(blockpath))
    #exit()
    
    nplogfilename = blockpath + "/delayedfeedback.pkl"
    nplog = NPLog.from_file(nplogfilename)
    print("Arrays logged: {}".format(nplog.get_names()))
    #exit()
    
    #channels = ['TimedMarkerPositionProvider.markerposition', 
    #            'AffineDisturbanceInducer.triggered', 
    #            'ScreenCalibratedMarkerPositionProvider.markerposition', 
    #            'WaitMarkerPosition_params', 
    #            'WaitMarkerPosition.renderpointposition', 
    #            'Trial.params', 
    #            'AffineDisturbanceInducer.markerposition', 
    #            'Experiment.params']

    mpt, mp = nplog.stack("TimedMarkerPositionProvider.markerposition")
    d = np.squeeze(mp)[:, [1, 2]]
    xmin, xmax = np.nanmin(d[:, 0]), np.nanmax(d[:, 0])
    ymin, ymax = np.nanmin(d[:, 1]), np.nanmax(d[:, 1])
    print("data shape: {}".format(d.shape))
    print("xmin, ymin: {}, {}".format(xmin, ymin))
    print("Size: {}, {}".format((xmax-xmin), (ymax-ymin)))
    print("Ratio: {}".format((xmax-xmin)/(ymax-ymin)))       
            
    plt.plot(d[:, 0], d[:, 1])
    plt.axis('equal')
    plt.show()
    exit()

    records = nplog.select_by_name("Trial.params")
    #print records
    #print [record.value.disturbance_mode for record in records]
    #for record in records:
    #    if record.value.disturbance_mode == 2 and record.value.disturbance_value < -0.5:
    #        print record.created

    #print nplog.events
    block = Block(nplog)
    print(block.params)
    for trial in block.trials:
        print(trial.t_start, trial.t_end)
        print([(event.created, event.msg) for event in trial.events if event.msg[:6]!= "State:"])

    block.plot_trials(prefix="block-{}-".format(iblock))

    #t, x = nplog.stack("WaitMarkerPosition.renderpointposition")
    #x = np.squeeze(x)  # only one marker
    #print(t.shape, x.shape)    
    #plt.plot(t, x, ".")
    #plt.show()
    #plt.plot(x[:, 0], x[:, 1], ".", alpha=0.2)
    #plt.show()
    
    #for name in nplog.get_names():
    #    records = nplog.select_by_name(name)
    #    #print("{} records: {}".format(name, records))
    #    #print("{}: {}".format(name, nplog.stack(name)))

    #for event in nplog.events:
    #    print("[{}]{}".format(event.relativecreated, event.msg))

    #trialparams = nplog.select_by_name("Trial.params")
    #for trialparam in trialparams:
    #    pp.pprint(vars(trialparam.value))


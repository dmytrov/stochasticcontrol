import os
import numpy as np
import matplotlib.pyplot as plt
import states.utils as su
import analysis.delayedfeedback.regression as re
import analysis.delayedfeedback.patherrors as pe
import analysis.delayedfeedback.trial as tr
import analysis.delayedfeedback.datalayer as dl
import analysis.delayedfeedback.database as db


if __name__ == "_main__":
    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected

    subjectname = "B"
    # Trials with disturbances
    dbevents = di.dbsession.query(db.TrialEvent) \
            .join(db.Trial, db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.TrialEvent.event_type_id == 2) \
            .all()
    
    print("Number of events: {}".format(len(dbevents)))
    for dbevent in dbevents:
        print(dbevent.time, dbevent.trial.number)
    #print("Number of trials: {}".format(len(dbtrials)))


    #trials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]

    #for dbtrial in dbtrials:
    #    print(dbtrial.number)



if __name__ == "__main__":
    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected

    disturbancemodes = [x[0] for x in di.dbsession.query(db.Trial.disturbance_mode).distinct().all()]
    feedback_delays = [x[0] for x in di.dbsession.query(db.Trial.feedback_delay).distinct().all()]
            
    subjectname = "B"
    
    # Trials with disturbances
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .order_by(db.Trial.number) \
            .all()
            #.filter(db.Trial.feedback_delay == -0.0) \
            #.filter(db.Trial.disturbance_mode == su.DisturbanceMode.Rotation) \
            #.order_by(db.Trial.number) \
    
    print("Number of trials: {}".format(len(dbtrials)))

    trials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]

    pttargets = tuple(set((trial.pttarget for trial in trials)))
    
    for feedback_delay in feedback_delays:
        for disturbancemode in disturbancemodes:
            for pttarget in pttargets:
                disturbancevalues = tuple(set((trial.disturbancevalue for trial in trials \
                        if trial.disturbancemode == disturbancemode and \
                            trial.feedbackdelay == feedback_delay and \
                            trial.pttarget == pttarget)))
                
                ntrials = np.zeros(len(disturbancevalues))
                for trial in trials:
                    if trial.disturbancemode == disturbancemode and \
                            trial.feedbackdelay == feedback_delay and \
                            trial.pttarget == pttarget:
                        i = disturbancevalues.index(trial.disturbancevalue)
                        ntrials[i] += 1
                print("feedback_delay: {}, disturbancemode: {}, pttarget: {}, counts:{}".format(
                        feedback_delay, disturbancemode, pttarget, ntrials))
                

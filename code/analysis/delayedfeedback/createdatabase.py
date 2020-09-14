from __future__ import print_function
import os
import sys
import pickle
import json
import datetime
from collections import namedtuple
import numpy as np
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy_utils import drop_database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import analysis.delayedfeedback.database as db
from utils.logger import *
from states.common import *



def print_list(info, lst):
    print(info)
    for item in lst:
        print("\t \"{}\"".format(item))



def object_to_record(obj):
    if isinstance(obj, np.ndarray):
            res = obj.tolist()
    elif hasattr(obj, "__dict__"):
            res = {k: object_to_record(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
            res = {k: object_to_record(v) for k, v in obj.items()}
    elif isinstance(obj, list):
            res = [object_to_record(i) for i in obj]
    else:
        res = obj
    return res


class EventType(object):
    BlockSequenceStart = 0
    BlockSequenceEnd = 1
    TrialSequenceStart = 2
    TrialSequenceEnd = 3
    TrialGoSignal = 4
    TrialGoalTimeout = 5
    TrialGoalReached = 6
    TrialStartReached = 7
    TrialDisturbanceTrigger = 8
    _id_count = 9

    
    messages = (
        (BlockSequenceStart, "Block: sequence start"),
        (BlockSequenceEnd, "Block: sequence end"),
        (TrialSequenceStart, "Trial: sequence start"),
        (TrialSequenceEnd, "Trial: sequence end"),
        (TrialStartReached, "Trial: start reached"),
        (TrialGoSignal, "Trial: go signal"),
        (TrialGoalTimeout, "Trial: goal timeout"),
        (TrialGoalReached, "Trial: goal reached"),
        (TrialDisturbanceTrigger, "Trial: disturbance trigger")
    )


    message_to_id = None 


    @classmethod
    def _preinit(cls):
        cls.message_to_id = {message[1]: message[0] for message in EventType.messages}
        

    @classmethod
    def fill_database(cls, dbsession):
        try:
            print("Adding event types... ", end="")
            for message in cls.messages:
                dbsession.add(db.EventType(id=message[0] ,desc=message[1]))
            dbsession.commit()
            print("OK")
        except IntegrityError:
            print("IntegrityError, skipped")
            dbsession.rollback()
        except:
            print("FAIL")
            dbsession.rollback()
            raise
        

EventType._preinit()    




def create_database(db_url):
    try:
        print("Creating new database [{}]... ".format(db_url), end="")
        try:
            drop_database(db_url)
        except:
            pass

        engine = create_engine(db_url)
        db.Base.metadata.create_all(engine)
        print("OK")
    except:
        print("FAIL")
        raise
    return engine


def open_database(db_url):
    engine = create_engine(db_url)
    return engine



def add_subject(dbsession, name, age, sex):
    try:
        print("Adding new subject [{}]... ".format(name), end="")
        dbsession.add(db.Subject(name=name, age=age, sex=sex))
        dbsession.commit()
        print("OK")
    except IntegrityError:
        print("IntegrityError, skipped")
        dbsession.rollback()
    except:
        print("FAIL")
        dbsession.rollback()
        raise

def _correct_nplog_events_order(nplog):
    events = []
    for event in nplog.events:
        if event.msg == "Trial: goal reached":
            events.insert(-2, event)
        else:
            events.append(event)
    nplog.events = events
        


def get_trial_start_stop(trial):
    t0, t1 = None, None
    for event in trial.events:
        if event.event_type_id == EventType.TrialSequenceStart:
            t0 = event.time
        elif event.event_type_id == EventType.TrialSequenceEnd:
            t1 = event.time
    #print(t0, t1)
    return t0, t1


def find_trials(block, times):
    trials_times = [(trial, get_trial_start_stop(trial)) for trial in block.trials]
    def find_trial(t):
        for trial, (t0, t1) in trials_times:
            #print(t0, t, t1)
            if t0 <= t and t <= t1:
                return trial
        return None
    return [find_trial(t) for t in times]



def add_session(dbsession, name, sessionpath):
    """ Add one experiment session to the database.
    Trial events:
     - trial start (go to home position)
     - go (target appears)
     - disturbance introduced
     - target reached or timeout
     - trial end
    Block events:
     - block start
     - block end
    Session events:
     - sesion start
     - session end
    """
    try:
        session = None  # current session
        block = None  # current block
        trial = None  # current trial
            
        # Find the subject
        print("Searching subject [{}]... ".format(name), end="")
        subj = dbsession.query(db.Subject).filter(db.Subject.name == name).first()
        if subj is None:
            raise ValueError("Subject [{}] is not found".format(name))
        print("OK")

        # Add session
        print("Session path is: \"{}\"".format(sessionpath))
        blockpaths = [x[0] for x in os.walk(sessionpath)][1:]
        print_list("Blocks found:", blockpaths)
        print("Addind new session for subject [{}]... ".format(name), end="")
        session = db.Session(subject=subj)
        dbsession.add(session)
        print("OK")

        # Process blocks
        for i, blockpath in enumerate(blockpaths):
            print("Loading block \"{}\"".format(blockpath))
            nplogfilename = os.path.join(blockpath, "delayedfeedback.pkl")
            nplog = NPLog.from_file(nplogfilename)
            #_correct_nplog_events_order(nplog)  # workaround for old recording of Gunnar

            dumps = lambda x: json.dumps(object_to_record(x))
            blockparamslist = nplog.select_by_name("Experiment.params")[0].value
            blockparamsliststr = dumps(blockparamslist)
            trialparamslist = [tp.value for tp in nplog.select_by_name("Trial.params")]
            #trialparamsliststrs = [dumps(tp) for tp in trialparamslist]
            
            print_list("Found logged streams:", nplog.get_names())
            block = db.Block(session=session, 
                number=blockparamslist.block_number, 
                paramslist=blockparamsliststr,
                opto_filename=os.path.join(blockpath, "REC-001.OPTO.npy"), 
                odau_filename=os.path.join(blockpath, "REC-001.ODAU.npy"))
            dbsession.add(block)
            
            # Find trial periods from ODAU sync channel.
            data = np.load(block.odau_filename)
            syncdata = data[:, 0]
            x = (syncdata > 0).astype(int)
            starts = np.where(np.diff(x) > 0)[0] + 1
            stops = np.where(np.diff(x) < 0)[0] + 1
            print("Found {} starts and {} stops in sync channel".format(len(starts), len(stops)))
            assert len(starts) == len(stops)

            print("Number of events: {}".format(len(nplog.events)))
            itrial = 0
            for event in nplog.events:
                # Process only trial events
                if isinstance(event, EventRecord):
                    if event.msg.startswith("Trial:"):
                        
                        if event.msg.startswith("Trial: sequence start"):
                            # Add new trial
                            trial = db.Trial(block=block, 
                                paramslist=dumps(trialparamslist[itrial]),
                                number=itrial,
                                disturbance_mode=trialparamslist[itrial].disturbance_mode,
                                feedback_delay=trialparamslist[itrial].feedback_delay,
                                opto_start=int(0.1 * starts[itrial]),
                                opto_stop=int(0.1 * stops[itrial]),
                                odau_start=starts[itrial],
                                odau_stop=stops[itrial],)
                            itrial += 1
                            dbsession.add(trial)

                        # Add trial event
                        trialevent = db.TrialEvent(trial=trial, 
                            event_type_id=EventType.message_to_id[event.msg], 
                            time=datetime.datetime.fromtimestamp(event.created))
                        dbsession.add(trialevent)
            
            # Trigger is logged at every frame when it is active.
            # Detect trigger onset and add a TrialEvent
            triggers = nplog.select_by_name("AffineDisturbanceInducer.triggered")
            triggerstimes = []  # trigger timestamps
            tprev = 0
            for record in triggers:
                if record.created > tprev + 0.1:
                    triggerstimes.append(datetime.datetime.fromtimestamp(record.created))
                tprev = record.created
            print("Number of disturbance triggers: {}".format(len(triggerstimes)))
            # Add trigger events
            trials = find_trials(block, triggerstimes)
            for trial, triggertime in zip(trials, triggerstimes):
                trialevent = db.TrialEvent(trial=trial, 
                        event_type_id=EventType.TrialDisturbanceTrigger,
                        time=triggertime)
                dbsession.add(trialevent)

            
                
        dbsession.commit()
        print("Commit OK")
    except:
        print("FAIL")
        raise


QUERY = """
SELECT trial_id1 as trial_id, time_start, time_stop FROM (
		(SELECT trial.id as trial_id1, trial_event.time as time_start FROM  trial_event LEFT JOIN trial ON trial.id = trial_event.trial_id 
		WHERE event_type_id = 2)
	INNER JOIN
		(SELECT trial.id as trial_id2, trial_event.time as time_stop FROM  trial_event LEFT JOIN trial ON trial.id = trial_event.trial_id 
		WHERE event_type_id = 3)
	ON trial_id1 = trial_id2
)
"""


if __name__ == "__main__":
    create = raw_input("Re-create database? (y/n)")
    if create == "y":
        db_url = 'sqlite:///delayed_feedback.db'
            
        engine = create_database(db_url)

        Session = sessionmaker(bind=engine)
        dbsession = Session()
        EventType.fill_database(dbsession)
        
        #add_subject(dbsession, "Gunnar Blohm", 40.5, "male")
        #add_session(dbsession, "Gunnar Blohm", sessionpath="../../../data/delayedfeedback/2017-06-12-(Gunnar)")

        name = "A"
        add_subject(dbsession, name, age=22.0, sex="male")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-12-(Felix-1)")

        name = "B"
        add_subject(dbsession, name, age=42.0, sex="male")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-21-(Dominik-2)")

        name = "C"
        add_subject(dbsession, name, age=25.0, sex="male")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-22-(Ben-3)")

        name = "D"
        add_subject(dbsession, name, age=25.0, sex="male")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-23-(Brandon-4)")

        name = "E"
        add_subject(dbsession, name, age=25.0, sex="male")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-24-(Deng-5)")

        name = "F"
        add_subject(dbsession, name, age=25.0, sex="male")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-25-(Jonathan-6)")
        
        name = "G"
        add_subject(dbsession, name, age=25.0, sex="female")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-29-(JVanderlinden-7)")

        name = "H"
        add_subject(dbsession, name, age=25.0, sex="female")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-30-(Johanna-lefthanded-8)")

        name = "I"
        add_subject(dbsession, name, age=22.0, sex="female")
        add_session(dbsession, name, sessionpath="../../../data/delayedfeedback/2017-08-31-(Ayako-9)")


        dbsession.close()




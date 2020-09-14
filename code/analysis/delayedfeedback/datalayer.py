""" Data access layer for the experiment.
Provides access to the the database and data stream files.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy_utils import drop_database
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
import analysis.delayedfeedback.database as db
import analysis.delayedfeedback.postprocess as pp


import functools
from inspect import signature

def memoize_all(func):
    """
    This is a caching decorator. It caches the function results for
    all the arguments combinations, so use it with care. It does not
    matter whether the arguments are passed as keywords or not.
    
    https://codereview.stackexchange.com/questions/78371/universal-memoization-decorator
    """
    cache = {}
    func_sig = signature(func)

    @functools.wraps(func)
    def cached(*args, **kwargs):
        bound = func_sig.bind(*args, **kwargs)
        key = frozenset(bound.arguments.items())

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return cached


class TrajectoryDataType(object):
    OfflineStream = 0  # offline data stored by optotrak
    Recovered = 1  # data recovered from the online stream used in the experiment
    ScreenProjected = 2  # screen projected recovered online data


class DataInterface(object):
    def __init__(self, db_url=None):
        if db_url is None:
            db_url = "sqlite:///" + os.path.join(os.path.dirname(__file__), "delayed_feedback.db")
            
        self.engine = create_engine(db_url)
        Session = sessionmaker(bind=self.engine)
        self.dbsession = Session()
        self._block_odau = {}  # block odau_filename to odau data dict
        self.data_type = TrajectoryDataType.Recovered

    @memoize_all
    def get_block_opto_data(self, block, data_type):
        if data_type == TrajectoryDataType.OfflineStream:
            data = np.load(block.opto_filename).astype(float)
            data[np.abs(data) > 1.0e6] = np.NAN  # optotrak is weird
            return data
        elif data_type == TrajectoryDataType.Recovered:
            blockpath = os.path.dirname(block.opto_filename)
            timestamps, trajectory = pp.read_restored_trajectory(blockpath)
            return trajectory
        elif data_type == TrajectoryDataType.ScreenProjected:
            blockpath = os.path.dirname(block.opto_filename)
            timestamps, trajectory = pp.read_projected_to_screen(blockpath)
            return trajectory

    @memoize_all
    def get_block_time_data(self, block):
        if self.data_type == TrajectoryDataType.Recovered:
            blockpath = os.path.dirname(block.opto_filename)
            timestamps, trajectory = pp.read_restored_trajectory(blockpath)
            return timestamps
        elif self.data_type == TrajectoryDataType.ScreenProjected:
            blockpath = os.path.dirname(block.opto_filename)
            timestamps, trajectory = pp.read_projected_to_screen(blockpath)
            return timestamps

    @memoize_all
    def get_block_odau_data(self, block):
        return np.load(block.odau_filename)
        

    def get_trial_opto_data(self, trial, data_type=None):
        if data_type is None:
            data_type = self.data_type
        data = self.get_block_opto_data(trial.block, data_type)[trial.opto_start:trial.opto_stop]
        return data


    def get_trial_time_data(self, trial):
        t = self.get_block_time_data(trial.block)[trial.opto_start:trial.opto_stop]
        return t


    def get_trial_odau_data(self, trial):
        data = self.get_block_odau_data(trial.block)[trial.odau_start:trial.odau_stop]
        return data
        


if __name__ == "__main__":
    di = DataInterface()
    trials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == "Felix Q").all()
    print("Number of trials: {}".format(len(trials)))
    
    f, (ax0, ax1, ax2) = plt.subplots(1, 3)
    for trial in trials[0:300]:
        #data = di.get_trial_odau_data(trial)
        #print(data.shape)
        #plt.plot(data)

        data = di.get_trial_opto_data(trial)
        print(data.shape)
        ax0.plot(data[:, 0])
        ax1.plot(data[:, 1])
        #ax2.plot(data[:, 2])
        ax2.plot(data[:, 1], data[:, 2])

    plt.show()

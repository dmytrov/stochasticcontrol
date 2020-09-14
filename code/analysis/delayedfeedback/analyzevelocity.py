import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import states.utils as su
import analysis.delayedfeedback.regression as re
import analysis.delayedfeedback.patherrors as pe
import analysis.delayedfeedback.trial as tr
import analysis.delayedfeedback.datalayer as dl
import analysis.delayedfeedback.database as db
import analysis.delayedfeedback.averagetrajectory as at
import analysis.delayedfeedback.plotutils as plu
import analysis.delayedfeedback.fittingutils as fu
import analysis.delayedfeedback.optimalcontrolutils as ocu
import analysis.delayedfeedback.deprecated.targetswitchingmodel as tsm


def plot_average(xys, tau=0.05, color="b"):
    """ Plots x/y data and regressed average with variance.
            xys : [(x, y)] - list of (x, y) tuples
    """
    if len(xys) == 0:
        return
    llr = re.LocallyLinearRegression(
            np.hstack([x for x, y in xys]),
            np.hstack([y for x, y in xys]),
            tau=0.05)
    
    x0 = np.linspace(np.min(llr.x), np.max(llr.x), 100)
    y0_mean_covar = [llr.regress(x) for x in x0]
    y0 = np.squeeze(np.array([m for m, c in y0_mean_covar]))
    #c = np.array([c for m, c in y0_mean_covar])
    c0 = np.squeeze(2*np.sqrt(np.array([c[0] for m, c in y0_mean_covar])))
    
    plt.plot(x0, y0, color=color)
    #print(x0, c0)
    plt.fill_between(x0, y0-c0, y0+c0, alpha=0.3)
    plt.plot(llr.x, llr.y[:, 0], ".", markersize=1, alpha=0.4, color=color)


def plot_single_trial(di, trial):
    
    print("Valid: {}".format(trial.isvalid))
    #if not trial.isvalid:
    #    return

    f = plt.figure(constrained_layout=True, figsize=(21, 7)) 
    gs = gridspec.GridSpec(ncols=3, nrows=3, figure=f)
    fax0 = f.add_subplot(gs[0, 0])
    fax1 = f.add_subplot(gs[1, 0])
    fax2 = f.add_subplot(gs[2, 0])

    ax1 = f.add_subplot(gs[:, 1])
    ax2 = f.add_subplot(gs[:, 2])

    # Plot 3D trajectory with recovered frames
    realtimetr = di.get_trial_opto_data(trial.dbtrial, data_type=dl.TrajectoryDataType.Recovered)
    fax0.plot(realtimetr[:, 0])
    fax1.plot(realtimetr[:, 1])
    fax2.plot(realtimetr[:, 2])


    x = trial.motiontrajectoryinterpolated
    v = np.linalg.norm(np.diff(trial.motiontrajectoryinterpolated, axis=0), axis=1)

    ax1.scatter(x[:-1, 0], x[:-1, 1], 
                marker="o", c=v, alpha=0.8, cmap=plt.cm.coolwarm)
    #ax1.plot(x[:, 0], x[:, 1], "o", alpha=0.3)
    
    ax1.axis("equal")
    ax1.set_title("trajectory")

    ax2.plot(v)
    ax2.set_title("velocity")

    plt.show()
    return

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    x = trial.motiontrajectoryinterpolated
    ax1.plot(x[:, 0], x[:, 1], "o", alpha=0.3)
    ax1.axis("equal")
    v = np.linalg.norm(np.diff(trial.motiontrajectoryinterpolated, axis=0), axis=1)
    ax2.plot(v)
    plt.show()


def plot_baseline_mean_var_by(subjectname, 
        feedback_delays=[-0.1], 
        disturbance_mode=su.DisturbanceMode.NoDisturbance,
        disturbance_values=[0.0],
        saveplots=False):

    print("Subject: {}".format(subjectname))
    
    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected

    # Select baseline trials
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == disturbance_mode) \
            .filter(db.Trial.feedback_delay == -0.0) \
            .all()
    baselinetrials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    validbaselinetrials = [t for t in baselinetrials if t.isvalid]
    n = len(baselinetrials)
    nvalid = len(validbaselinetrials)
    print("Valid baseline trials: {} of {}, {}%".format(nvalid, n, 100*nvalid/n))

    pttargets = tuple(set((trial.pttarget for trial in baselinetrials)))

    psize=3
    fig = plt.figure(constrained_layout=True, figsize=(5*psize, 3*psize)) 
    gs = gridspec.GridSpec(ncols=5, nrows=4, figure=fig)

    for icol, pttarget in zip(range(len(pttargets)), pttargets):
        faxtrajectory = fig.add_subplot(gs[0, icol])
        xys = []
        for trial in validbaselinetrials:
            if trial.pttarget == pttarget:
                xys.append((trial.phase, trial.motiontrajectoryinterpolated))
        plu.plot_average(xys, ax=faxtrajectory)
        plt.title("Trajectory mean (N={})".format(len(xys)))
        
        faxvelocity = fig.add_subplot(gs[1, icol])
        xys = []
        for trial in validbaselinetrials:
            if trial.pttarget == pttarget:
                x = trial.phase[:-1]
                # Velocity
                y = np.linalg.norm(np.diff(trial.motiontrajectoryinterpolated, axis=0), axis=1)
                xys.append((x, y))
        plu.plot_average(xys, ax=faxvelocity)
        
            
        if len(xys) > 0:
            xys = [xys[0]]
            # Polynomial fit
            ordercolors = ((2, "c--"), (3, "g--"), (4, "k--"), (5, "r--"))
            for order, color in ordercolors:
                A, ystar = fu.fit_ploynomial(np.hstack([x for x, y in xys]), np.hstack([y for x, y in xys]), order=order)
                f = fu.Polynomial(A)
                x = np.linspace(0, 1, 100)
                faxvelocity.plot(x, f(x), color)
        
        plt.title("Velocity mean")
        faxvelocity.set_xlabel("phase")

        # Predict target from ballitsic movement
        faxprediction = fig.add_subplot(gs[2, icol])
        plu.plot_average(xys, ax=faxprediction)
        if len(xys) > 0:
            maxphase = 0.1
            def phases(x):  # selects phases of interest
                s = np.logical_or(np.logical_or(x<np.min(x)+0.3*maxphase, x>np.max(x)-maxphase),
                        np.logical_and(x>0.5-0.5*maxphase, x<0.5+0.5*maxphase))
                n = 2
                s = np.zeros(len(x), dtype=bool)
                a = np.argmax(x>0.05)
                b = np.argmax(x>0.95)
                m = np.argmax(x>0.5)
                s[a:a+2*n] = True
                s[m-n:m+n] = True
                s[b-2*n:b] = True
                return s
            xys = [(x[phases(x)], y[phases(x)]) for x, y in xys]
            for x, y in xys:
                faxprediction.plot(x, y, "r*")
            # Polynomial fit
            ordercolors = ((2, "c--"), (3, "g--"), (4, "k--"))
            for order, color in ordercolors:
                A, ystar = fu.fit_ploynomial(np.hstack([x for x, y in xys]), np.hstack([y for x, y in xys]), order=order)
                f = fu.Polynomial(A)
                x = np.linspace(0, 1, 100)
                faxprediction.plot(x, f(x), color)

        # Polynomial + exponential fit
        fax = fig.add_subplot(gs[3, icol])
        fax.axis("equal")
        trials = [trial for trial in validbaselinetrials if trial.pttarget == pttarget]
        ocu.fit_trajectory_nonesense(trials, ax=fax)
        


    fig.suptitle("Subject: {}, disturbance mode: {}, disturbance values: {}".format(
            subjectname, disturbance_mode, disturbance_values))
    plt.show()


def plot_by(subjectname, 
        feedback_delays=[-0.1], 
        disturbance_mode=su.DisturbanceMode.Rotation,
        disturbance_values=[0.0],
        saveplots=False):

    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected

    print("Subject: {}".format(subjectname))
    # Select baseline trials
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == su.DisturbanceMode.NoDisturbance) \
            .filter(db.Trial.feedback_delay == -0.0) \
            .all()
    baselinetrials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    n = np.sum([t.isvalid for t in baselinetrials])
    print("Valid baseline trials: {} of {}, {}%".format(n, len(baselinetrials), 100*n/len(baselinetrials)))

    pttargets = tuple(set((trial.pttarget for trial in baselinetrials)))
    #print(pttargets)

    # Plot baseline trials trajectories
    #for trial in baselinetrials:
    #    plot_single_trial(di, trial)

    # Trials with disturbances
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == disturbance_mode) \
            .filter(db.Trial.feedback_delay == feedback_delays[0]) \
            .all()
    disturbtrials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    n = np.sum([t.isvalid for t in disturbtrials])
    print("Valid disturbed trials: {} of {}, {}%".format(n, len(disturbtrials), 100*n/len(disturbtrials)))

    #return
    
    # Plot disturbed trials trajectories
    #for trial in disturbtrials:
    #    x = trial.motiontrajectoryinterpolated
    #    plt.plot(x[:, 0], x[:, 1])
    #plt.show()
    
    for pttarget in pttargets:
        xys = []
        for trial in baselinetrials:
            if trial.pttarget == pttarget:
                x = trial.phase[:-1]
                y = np.linalg.norm(np.diff(trial.motiontrajectoryinterpolated, axis=0), axis=1)
                x = np.arange(0, len(y))
                xys.append((x, y))
                #plt.plot(x, y, color="blue")
        #plot_average(xys)

        for trial in disturbtrials:
            #print(trial.disturbancevalue)
            if trial.pttarget == pttarget and \
                    trial.disturbancevalue not in disturbance_values:
                x = trial.phase[:-1]
                y = np.linalg.norm(np.diff(trial.motiontrajectoryinterpolated, axis=0), axis=1)
                x = np.arange(0, len(y))
                plot_single_trial(di, trial)
                #plt.plot(x, y)
        
        #plt.show()
        

def create_disturbances_dict(di):
    assert isinstance(di, dl.DataInterface)
    disturbancemodes = [x[0] for x in di.dbsession.query(db.Trial.disturbance_mode).distinct().all()]
    feedback_delays = [x[0] for x in di.dbsession.query(db.Trial.feedback_delay).distinct().all()]
    disturbancevalues = {}
    # Trials with disturbances
    dbtrials = di.dbsession.query(db.Trial).all()
    trials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    for disturbancemode in disturbancemodes:
        disturbancevalues = tuple(set((trial.disturbancevalue for trial in trials \
                        if trial.disturbancemode == disturbancemode and \
                            trial.feedbackdelay == feedback_delay and \
                            trial.pttarget == pttarget)))
    



def plot_switching_sigmoid(subjectname):
    dirpath = "./plots/"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    feedback_delays = [-0.0]
    disturbance_mode = su.DisturbanceMode.NoDisturbance
    disturbance_values = [0.0]
    saveplots = False

    print("Subject: {}".format(subjectname))
    
    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected

    # Select baseline trials
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == su.DisturbanceMode.NoDisturbance) \
            .filter(db.Trial.feedback_delay == -0.0) \
            .all()
    baselinetrials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    validbaselinetrials = [t for t in baselinetrials if t.isvalid]
    n = len(baselinetrials)
    nvalid = len(validbaselinetrials)
    print("Valid baseline trials: {} of {}, {}%".format(nvalid, n, 100*nvalid/n))

    # Select translation disturbed trials
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == su.DisturbanceMode.Translation) \
            .filter(db.Trial.feedback_delay == -0.0) \
            .all()
    disturbtrials = [tr.Trial(di, dbtrial) for dbtrial in dbtrials]
    validdisturbtrials = [t for t in disturbtrials if t.isvalid]
    n = len(disturbtrials)
    nvalid = len(validdisturbtrials)
    print("Valid disturbed trials: {} of {}, {}%".format(nvalid, n, 100*nvalid/n))
    

    validbaselinetrials = validbaselinetrials[:]
    validdisturbtrials = validdisturbtrials[:]
    straight_trials = [(trial.motiontrajectoryinterpolated[trial.a:trial.b], trial.ptstart, trial.pttarget) for trial in validbaselinetrials]
    disturbed_trials = [(trial.motiontrajectoryinterpolated[trial.a:trial.b], trial.ptstart, trial.pttarget, trial.ptstartcorrected, trial.pttargetcorrected) for trial in validdisturbtrials]
    
    regressortypes = [ \
            tsm.RegressorType.Velocity1D, 
            tsm.RegressorType.Acceleration, 
            tsm.RegressorType.OptimalTarget, 
            tsm.RegressorType.OptimalTrajectory,
            tsm.RegressorType.Const,
            ]
    
    if True:
        plt.figure()
        trials = validbaselinetrials + validdisturbtrials
        for i, trial in zip(range(len(trials)), trials):
            trajectory = trial.motiontrajectoryinterpolated
            plt.plot(trajectory[:, 0], trajectory[:, 1])
            plt.plot(trial.pttarget[0], trial.pttarget[1], "xb")
            plt.plot(trial.pttargetcorrected[0], trial.pttargetcorrected[1], "or")
        plt.axis("equal")
        plt.savefig("./plots/switching-training.pdf")
        plt.close()
            
    # Learn linear policy and noise from the trining trials
    policy, sigm_as, sigm_bs, noise, trajectories = tsm.fit_target_switching(straight_trials, [], 
            regressortypes=regressortypes, 
            fitnoise=True,
            fitpolicy=True,
            fitsigmoid=True,
            maxiter=1000)

    # Sample a trajectory from the policy
    for i, straight_trial in zip(range(len(straight_trials)), straight_trials):
        trajectory = straight_trial[0]
        x = tsm.run_policy(policy, trajectory[:3], trajectory[-1], n=len(trajectory), regressortypes=regressortypes)
        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.plot(x[:, 0], x[:, 1])
        plt.axis("equal")
        plt.savefig("./plots/switching-generated-({}).pdf".format(i))
        plt.close()

    # Plot the switching target sigmoids
    for i, disturbed_trial in zip(range(len(disturbed_trials)), disturbed_trials):
        # Fit the switching sigmoids
        _, sigm_as, sigm_bs, _, trajectories = tsm.fit_target_switching([], [disturbed_trial], 
                initialpolicy=policy,
                initialnoise=noise,
                regressortypes=regressortypes, 
                fitnoise=False,
                fitpolicy=False,
                fitsigmoid=True,
                maxiter=1000)
        sigma_a = sigm_as[0]
        sigma_b = sigm_bs[0]
        trajectory, _, ti, _, tc = disturbed_trial
        sigmoid = tsm.sigmoidal(sigma_a, sigma_b, np.linspace(0, 1, len(trajectory)))
        
        psize = 5
        fig = plt.figure(constrained_layout=True, figsize=(3*psize, 1*psize)) 
        gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
        
        fax = fig.add_subplot(gs[0, 0])
        fax.scatter(trajectory[:, 0], trajectory[:, 1], 
                marker="o", c=sigmoid, alpha=0.8, cmap=plt.cm.coolwarm)
        fax.plot(ti[0], ti[1], "xb")
        fax.plot(tc[0], tc[1], "or")
        plt.axis("equal")

        fax = fig.add_subplot(gs[0, 1])
        fax.plot(trajectory[:, 0])
        fax.plot(trajectory[:, 1])
        fax.plot(sigmoid)

        fax = fig.add_subplot(gs[0, 2])
        negloglik = [[-tsm.fit_target_switching([], [disturbed_trial], 
                initialpolicy=policy,
                initialnoise=noise,
                initialsigmoid=[[i_sigma_a],[i_sigma_b]],
                regressortypes=regressortypes, 
                fitnoise=False,
                fitpolicy=False,
                fitsigmoid=False,
                loglikonly=True)
                    for i_sigma_a in np.linspace(1.0, 30.0, 50)]
                        for i_sigma_b in np.linspace(0.1, 1.0, 50)]
  
        fax.axis([1.0, 30.0, 0.1, 1.0])
        fax.imshow(negloglik, extent=[1.0, 30.0, 0.1, 1.0], origin="lower", aspect="auto")
        fax.plot(sigma_a, sigma_b, "xr")
        fax.set_title("sigm_a={}, sigm_b={}".format(sigma_a, sigma_b))
        fax.set_xlabel("a")
        fax.set_ylabel("b")
        plt.savefig("./plots/switching-fitted-({}).pdf".format(i))
        plt.close()
    


    
    


if __name__ == "__main__":
    subjects = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    for subject in ["I"]:
        plot_switching_sigmoid(subject)
    exit()

    for subject in subjects:
        plot_baseline_mean_var_by(subject)
    exit()


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
                

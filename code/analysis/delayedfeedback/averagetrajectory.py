import os
import numpy as np
import matplotlib.pyplot as plt
import states.utils as su
import analysis.delayedfeedback.regression as re
import analysis.delayedfeedback.patherrors as pe
import analysis.delayedfeedback.trial as tr
import analysis.delayedfeedback.datalayer as dl
import analysis.delayedfeedback.database as db



def plot_mean_var(x, covar, color="b"):
    """ Plot mean line and confidence intervals.
            x : [N, 2]
            covar : [N] - variance along the line
    """
    assert len(x) == len(covar)
    # Compute normals to the mean line
    dx = np.diff(x, axis=0)
    dx = np.vstack([dx, dx[-1]])
    dx = dx / np.linalg.norm(dx, axis=-1)[:, np.newaxis]
    mnormal = np.array([[0, 1], [-1, 0]])
    dxnormal = dx.dot(mnormal)
    if len(covar.shape) == 2:
        # Covariance has values for X and Y.
        # Find covariance in normal direction
        vbasis = np.stack([dx, dxnormal], axis=2)
        vbasisinv = np.linalg.inv(vbasis)
        covar_global = np.array([basis.T.dot(np.diag(c)).dot(basis) for c, basis in zip(covar, vbasis)])
        covar = covar_global[:, 1, 1]
    var = np.sqrt(covar)

    xleft = x - dxnormal * var[:, np.newaxis]
    xright = x + dxnormal * var[:, np.newaxis]

    polyline = np.vstack([xleft, np.flip(xright, axis=0)])
    p = plt.Polygon(polyline, closed=True, fill=True, alpha=0.2, color=color)
    plt.plot(x[:, 0], x[:, 1], color=color)
    plt.gca().add_patch(p)
   

#plot_mean_var(np.array([[0, 0], [1, 0], [2, 0], [3, 1], [3, 2], [3, 3]]), 
#        np.array([[0.1, 0.3], [0.2, 0.2], [0.3, 0.1], [0.1, 0.3], [0.2, 0.2], [0.3, 0.1]]))
#plt.show()
#exit()


def plot_average_trajectory(trials, color="b"):
    if len(trials) == 0:
        return
    x0 = np.linspace(0, 1.0, 100)
    #phases = pe.select_valid_phase(motiontrajectories, phases)
    #motiontrajectories, phases, ptstarts, ptends = [a, b, c, d 
    #        for a, b, c, d in zip(motiontrajectories, phases, ptstarts, ptends)
    #        if np.isnan(a ]
    #motiontrajectories = [tr.interpolate_missing_data(trajectory) for trajectory in motiontrajectories]
    #motiontrajectories = [pr.resample_trajectory_normalize_path(trajectory, len(x0)) for trajectory in motiontrajectories]
    
    #a = np.hstack([trial.phasecorrected[trial.a:trial.b] for trial in trials])
    #b = np.vstack([trial.motiontrajectoryinterpolated[trial.a:trial.b] for trial in trials])
    #print(a.shape, b.shape)
    llr = re.LocallyLinearRegression(
            np.hstack([trial.phasecorrected[trial.a:trial.b] for trial in trials]),
            np.vstack([trial.motiontrajectoryinterpolated[trial.a:trial.b] for trial in trials]),
            tau=0.05)
    
    y0_mean_covar = [llr.regress(x) for x in x0]
    y0 = np.array([m for m, c in y0_mean_covar])
    
    c = np.array([c for m, c in y0_mean_covar])  # [N, 2], global coordinates

    #c0 = 2*np.sqrt(np.array([c[0] for m, c in y0_mean_covar]))
    
    plot_mean_var(y0, c, color)
    #plt.plot(llr.x, llr.y[:, 0], ".")
    #plt.plot(x0, y0[:, 0], "r")
    #plt.axis("equal")
    #plt.show()
    
    #plt.plot(llr.y[:, 0], llr.y[:, 1])
    plt.plot(llr.y[:, 0], llr.y[:, 1], ".", markersize=1, alpha=0.4, color=color)
    #plt.plot(x0, 10*np.sqrt(c[:, 0]))
    #plt.plot(x0, 10*np.sqrt(c[:, 1]))
    #plt.fill_between(x0, y0-c0, y0+c0, alpha=0.3)
    #plt.plot(y0[:, 0], y0[:, 1], "r")
    plt.axis("equal")
    #plt.show()


    
def plot_by_subject_delay(subjectname, 
        feedback_delays=[-0.0], 
        disturbance_mode=su.DisturbanceMode.NoDisturbance,
        saveplots=False):

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
    
    # Construct llr predictors for directional errors
    x0s = sorted(list(set([trial.pttarget[0] for trial in baselinetrials])))
    llrs = {}
    for x0 in x0s:
        x0trials = [trial for trial in baselinetrials if trial.isvalid and trial.pttarget[0] == x0]
        print(len(x0trials))
        phases = np.hstack([trial.phasecorrected[trial.a:trial.b-1] for trial in x0trials])
        #motions = np.vstack([trial.motiontrajectoryinterpolated[trial.a:trial.b] for trial in x0trials])
        disterrs = np.hstack([pe.raydistance_error(trial.motiontrajectoryinterpolated[trial.a:trial.b], trial.ptstart, trial.pttarget) for trial in x0trials])
        llrs[x0] = re.LocallyLinearRegression(phases, disterrs)
    
    # Trials with disturbances
    dbtrials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == disturbance_mode) \
            .all()
    print("Number of trials: {}".format(len(dbtrials)))
    trials = [tr.Trial(di, dbtrial, baselinellr=llrs[dbtrial.params["goal_position"][0]]) for dbtrial in dbtrials]

    disturbances = sorted(list(set([np.sum(trial.disturbancevalue) for trial in trials])))

    i = 0
    for x0 in x0s:
        for disturbance in disturbances:
            plt.figure(figsize=[8, 8])
            colors = ["k", "r", "g", "b"]
            
            x0trials = [trial for trial in baselinetrials if trial.isvalid and trial.pttarget[0] == x0]
            plot_average_trajectory(x0trials, color=colors.pop(0))

            for feedback_delay in feedback_delays:
                trials_selected = [trial for trial in trials 
                        if trial.isvalid and \
                            trial.pttarget[0] == x0 and \
                            np.abs(np.sum(trial.disturbancevalue) - disturbance) < 0.01 and \
                            trial.feedbackdelay == feedback_delay]
                print("target[0]: {}, disturbance: {}, selected trials: {}".format(x0, disturbance, len(trials_selected)))
                plot_average_trajectory(trials_selected, color=colors.pop(0))
                
                if len(trials_selected) > 0:
                    pt = trials_selected[0].pthome
                    plt.plot(pt[0], pt[1], ".", markersize=10)
                    pt = trials_selected[0].pttarget
                    plt.plot(pt[0], pt[1], ".", markersize=10)
                    pt = trials_selected[0].pttargetcorrected
                    plt.plot(pt[0], pt[1], ".", markersize=10)

            dm = ["none", "transl", "rot"][disturbance_mode]
            plt.title("Subject ({}). Disturbance({})".format(subjectname, dm))
            if saveplots:
                i += 1
                plt.savefig(os.path.join("./plots/mean-trajectories_subj({})_dist({})_cond({}).pdf".format(
                        subjectname, dm, i)))
                plt.close()
            else:
                plt.show()
            


if __name__ == "__main__":
    subjectname = "I"
    saveplots = True
    feedback_delays = [-0.0, -0.1, -0.2]
    for disturbance_mode in [su.DisturbanceMode.Rotation]:
        plot_by_subject_delay(subjectname, feedback_delays, disturbance_mode, saveplots)
            
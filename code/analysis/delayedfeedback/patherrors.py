import matplotlib.pyplot as plt
import numpy as np
import states.utils as su




def directional_error(motiontrajectory, ptstart, ptend):
    """ Computes directional error (signed angle)
    """
    assert len(motiontrajectory.shape) == 2
    assert motiontrajectory.shape[1] == 2
    
    mv = np.diff(motiontrajectory, axis=0)  # motion vector
    md = mv / np.linalg.norm(mv, axis=1)[:, np.newaxis]  # motion direction
    vgoal = (ptend - motiontrajectory) / np.linalg.norm(ptend - motiontrajectory, axis=1)[:, np.newaxis]
    
    direrr = np.arctan2(vgoal[:-1, 1], vgoal[:-1, 0]) - np.arctan2(md[:, 1], md[:, 0])
    direrr[direrr > np.pi] -= 2*np.pi
    direrr[direrr < -np.pi] += 2*np.pi
    return direrr


def raydistance_error(motiontrajectory, ptstart, ptend):
    """ Computes ray-target distance error (signed).
        disterr:
            positive for target right, 
            negatve for target left
    """
    assert len(motiontrajectory.shape) == 2
    assert motiontrajectory.shape[1] == 2
    
    mv = np.diff(motiontrajectory, axis=0)  # motion vector
    md = mv / np.linalg.norm(mv, axis=1)[:, np.newaxis]  # motion direction
    vgoal = (ptend - motiontrajectory)  # target vector, unnormatized
    mdrot = np.vstack([md[:, 1], -md[:, 0]]).T  # md rotated 90 degrees
    
    disterr = np.sum(mdrot * vgoal[:-1, :], axis=1) 
    disterr = disterr / np.linalg.norm(ptstart - ptend)  # normalize

    #phase = normalized_motion_phase(motiontrajectory, ptstart, ptend)
    #plt.plot(phase[:-1], disterr)
    #plt.show()

    return disterr



def compute_trial_error(
        motiontrajectory, times,
        ptstart, ptend, 
        disturbancemode, 
        disturbancevalue, 
        disturbanceonsettime=None,  # if none, reconstruct onset form the threshold ratio
        disturbance_threshold=9.0):
    """ Computes directional error for every time point.
        Returns:
            screentrajectory - actual presented screen trajectory including disturbances
            errors : [N]
            disturbance_onset_time
            disturbance_onset_index
    """
    assert len(motiontrajectory.shape) == 2
    assert motiontrajectory.shape[1] == 2
    assert len(motiontrajectory) == len(times)

    #err_func = directional_error
    err_func = raydistance_error
    
    if disturbancemode == su.DisturbanceMode.NoDisturbance:
        return motiontrajectory, err_func(motiontrajectory, ptstart, ptend), None, None

    if disturbanceonsettime is None:
        disttostart = np.linalg.norm(motiontrajectory - ptstart, axis=1)
        disttogoal = np.linalg.norm(motiontrajectory - ptend, axis=1)
        triggered = disttogoal < disturbance_threshold * disttostart
        i = np.argmax(triggered)
        disturbanceonsettime = times[i]

    # Rotation or translation disturbance inverse transformation
    mtrans = None
    if disturbancemode == su.DisturbanceMode.Rotation:
        mtrans = su.from_rotation_around(-disturbancevalue, ptstart)
    elif disturbancemode == su.DisturbanceMode.Translation:
        mtrans = su.from_translation(-np.array(disturbancevalue))
    screentrajectory = su.apply_disturbance(mtrans, motiontrajectory)
        
    # Compute directional errors
    errors1 = err_func(motiontrajectory, ptstart, ptend)
    errors2 = err_func(screentrajectory, ptstart, ptend)
    
    # Combine two errors at the point where disturbance was introduced
    i = np.argmax(times >= disturbanceonsettime)
    return  np.vstack([motiontrajectory[:i], screentrajectory[i:]]), \
            np.hstack([errors1[:i], errors2[i:]]), \
            disturbanceonsettime, i





def compute_phases(motiontrajectories, ptstarts, ptends):
    assert len(motiontrajectories) == len(ptstarts)
    assert len(motiontrajectories) == len(ptends)

    phases = [normalized_motion_phase(screentrajectory, screentrajectory[0], ptend) \
            for screentrajectory, ptstart, ptend in zip(motiontrajectories, ptstarts, ptends)]

    return phases


def compute_path_error_mean_var(motiontrajectories, ptstarts, ptends):
    assert len(motiontrajectories) == len(ptstarts)
    assert len(motiontrajectories) == len(ptends)

    #err_func = directional_error
    err_func = raydistance_error
    
    # Compute prajectory error
    errs = [err_func(screentrajectory, screentrajectory[0], ptend) \
            for screentrajectory, ptstart, ptend in zip(motiontrajectories, ptstarts, ptends)]
    phases = [normalized_motion_phase(screentrajectory, screentrajectory[0], ptend) \
            for screentrajectory, ptstart, ptend in zip(motiontrajectories, ptstarts, ptends)]

    return errs, phases



def select_valid_phase(data, phase, phasemin=0.02, phasemax=0.98):
    valid_start = np.argmax(phase[:-1] >= phasemin)
    valid_end = valid_start + np.argmax(phase[valid_start:] > phasemax)
    if valid_start == valid_end:
        valid_end = len(data)
    return data[valid_start:valid_end], phase[valid_start:valid_end]
        


def select_valid_phases(datas, phases, phasemin=0.02, phasemax=0.98):
    valid_phases = []
    valid_datas = []
    for data, phase in zip(datas, phases):
        d, p = select_valid_phase(data, phase, phasemin, phasemax)
        valid_datas.append(d)
        valid_phases.append(p)
        
    return valid_datas, valid_phases





    
def select_valid_parts(errs, phases):
    valid_phases = []
    valid_errs = []
    for err, phase in zip(errs, phases):
        valid_start = np.argmax(np.logical_and(phase[:-1] >= 0.05, np.abs(err) < np.pi/4.0))
        valid_end = valid_start + np.argmax(np.abs(err[valid_start:]) > np.pi/4.0)
        if np.abs(err[valid_end]) < np.pi/4.0:
            valid_end = len(err)
        valid_phases.append(phase[valid_start:valid_end])
        valid_errs.append(err[valid_start:valid_end])
        
    return valid_phases, valid_errs

    
    
def plot_regressed(llr):
    x0 = np.linspace(0, 1, 100)    
    y0_mean_covar = [llr.regress(x) for x in x0]

    y0 = np.array([m for m, c in y0_mean_covar])
    c0 = 2*np.sqrt(np.array([c for m, c in y0_mean_covar]))
    plt.plot(llr.x, llr.y, ".", markersize=2)
    plt.fill_between(x0, y0-c0, y0+c0, alpha=0.3)
    plt.plot(x0, y0, "r")
    plt.show()






class ConditionalGaussian(object):

    def __init__(self, x):
        """ x[N, D] : N data points of D dimensions
        """
        invalid = np.any(np.isnan(x), axis=1)
        self.x = x[np.logical_not(invalid)]
        self.x_mean = np.mean(self.x, axis=0)
        self.x_centered = x - self.x_mean
        self.w = 1.0 / np.sqrt(np.sum(self.x_centered**2, axis=0)/len(self.x_centered))
        self.wx = self.w * self.x_centered
        # Covariance matrix
        self.wxtwx = np.dot(self.wx.T, self.wx)/len(self.wx)
        self.D = len(self.x_mean)
        #plt.imshow(self.wxtwx)
        #plt.show()


    def conditoin(self, xinds, xstar):
        # Compute p(y|x)
        yinds = [i for i in range(len(self.wxtwx)) if i not in xinds]
        xinds, yinds = np.array(xinds), np.array(yinds)
        cov_xy = self.wxtwx[xinds][:, yinds]
        cov_yx = self.wxtwx[yinds][:, xinds]
        cov_xx = self.wxtwx[xinds][:, xinds]
        cov_yy = self.wxtwx[yinds][:, yinds]

        decoder = np.dot(cov_yx, np.linalg.inv(cov_xx))
        decoder_covar = cov_yy - np.dot(cov_yx, np.linalg.inv(cov_xx).dot(cov_xy))

        wxstar = self.w[xinds] * (xstar - self.x_mean[xinds])
        wystar = decoder.dot(wxstar.T).T
        ystar = wystar / self.w[yinds] + self.x_mean[yinds]
        decoder_covar_unw = decoder_covar / np.dot(self.w[yinds, np.newaxis], self.w[np.newaxis, yinds])
        return ystar, decoder_covar_unw


def gaussian_pdf(pt, mean, covar):
    n = len(covar)
    z = np.sqrt(np.linalg.det(2*np.pi*covar))
    xc = pt - mean
    return 1.0 / z * np.exp(-0.5 * np.sum(xc[:, np.newaxis].dot(xc[np.newaxis ,:]) * np.linalg.inv(covar)))


class TargetPredictor(object):
    def __init__(self, motiontrajectories, ptstarts, ptends):
    
        phases = [normalized_motion_phase(motiontrajectory, motiontrajectory[0], ptend) \
                for motiontrajectory, ptstart, ptend in zip(motiontrajectories, ptstarts, ptends)]

        # Need:
        #    control: acceleration (vector)
        #    state: velocity (scalar), target (vector)

        # Basis change:
        #    global to local: inv(B).dot(v_global)
        #    local to global: B.dot(v_local)
        
        avts = []  # accel, velocity, target
        for x, phase, ptend in zip(motiontrajectories, phases, ptends):
            istart = np.argmax(phase > 0.1)
            x = x[istart:]

            v = np.diff(x, axis=0)  # velocity
            absv = np.linalg.norm(v, axis=1)  # absolute velocity
            vn = v / absv[:, np.newaxis]  # normalized velocity
            vnnormal = vn[:, [1, 0]] * np.array([1.0, -1.0])  # normal to velocity
            vbasis = np.stack([vn, vnnormal], axis=2)
            vbasisinv = np.linalg.inv(vbasis)
            a = np.diff(v, axis=0)  # acceleration vector

            v_local = np.einsum("kij,kj->ki", vbasisinv, vn)  # loval velocity
            a_local = np.einsum("kij,kj->ki", vbasisinv[1:], a)  # acceleration vector in velocity basis
            target_local = np.einsum("kij,kj->ki", vbasisinv, (np.array(ptend) - x[:-1]))
        
            avts.append(np.hstack([
                    a_local,  # acceleration [2] 
                    #np.vstack([a_local[:1], a_local[:-1]]),  # acceleration [2]
                    #np.vstack([a_local[1:], a_local[-1:]]),  # acceleration [2]
                    absv[1:, np.newaxis],  # velocity [1]
                    target_local[1:],  # target [2]
                    ]))
        
        self.avt = np.vstack(avts)
        ivalid = np.logical_not(np.any(np.isnan(self.avt), axis=1))
        self.avt = self.avt[ivalid]
        self.cg = ConditionalGaussian(self.avt)
        #print(np.any(np.isnan((self.avt))))
        #plt.imshow(self.cg.wxtwx)
        #plt.show()
        

    def predict_target(self, x, ptend=None, plotfilename=None):
        assert len(x) >= 3
        v = np.diff(x, axis=0)  # velocity
        absv = np.linalg.norm(v, axis=1)  # absolute velocity
        vn = v / absv[:, np.newaxis]  # normalized velocity
        
        vnnormal = vn[:, [1, 0]] * np.array([1.0, -1.0])  # normal to velocity
        vbasis = np.stack([vn, vnnormal], axis=2)
        vbasisinv = np.linalg.inv(vbasis)
        a = np.diff(v, axis=0)  # acceleration vector
        a_local = np.einsum("kij,kj->ki", vbasisinv[1:], a)  # acceleration vector in velocity basis

        target_predicted, covar = self.cg.conditoin(range(self.cg.D-2), 
                np.hstack([a_local, absv[1:, np.newaxis]]))
        print(target_predicted.shape)
        # Convert to global coordinates
        target_predicted_global = np.einsum("kij,kj->ki", 
                vbasis[1:], 
                target_predicted) + x[1:-1]
        covars_global = [basis.T.dot(covar).dot(basis) for basis in vbasis]
        probs = []
        for mean, covar in zip(target_predicted_global, covars_global):
            probs.append(gaussian_pdf(ptend, mean, covar))

        #plt.plot(target_predicted[:, 0], target_predicted[:, 1], "r")
        plt.figure()
        plt.scatter(target_predicted_global[:, 0], target_predicted_global[:, 1], 
                marker="o", c=probs, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.plot(target_predicted_global[:, 0], target_predicted_global[:, 1], alpha = 0.5)
        if ptend is not None:
            plt.plot(ptend[0], ptend[1], "o", color="r")
        plt.plot(x[:, 0], x[:, 1])
        # Plot segments
        s = [[a, b, [None, None]] for a, b in zip(x[:-1], target_predicted_global)]
        s = np.concatenate(s, axis=0)
        plt.plot(s[:, 0], s[:, 1], alpha=0.2)
        plt.axis("equal")
        plt.title("Predicted target")
        if plotfilename is not None:
            plt.savefig(plotfilename)
            plt.close()
        else:
            pass
            plt.show()
        plt.plot(probs)
        plt.show()
        return target_predicted, target_predicted_global, covar, covars_global, probs


    def target_probability(self, x, target):
        target_predicted, target_predicted_global, covar, covars_global, probs = self.predict_target(x, target)
        return probs


        



def detect_correction_onset(baseline_llr, 
        motiontrajectory, times,
        ptstart, ptend, 
        disturbancemode, 
        disturbancevalue, 
        disturbanceonsettime=None,  # if none, reconstruct onset form the threshold ratio
        disturbance_threshold=9.0,
        plotfilename=None):
    """Threshold is defined as an event whenthe direction error is more than 3*sigma
       and decreases into the correction direction.
    """

    # Error considering the disturbance
    screentrajectory, err_screen, disturbancetime, disturbanceindex = compute_trial_error(
        motiontrajectory, times,
        ptstart, ptend, 
        disturbancemode, 
        disturbancevalue, 
        disturbanceonsettime,
        disturbance_threshold)
    
    # Error as if there is no disturbance (ballistic)
    err_ballistic = raydistance_error(motiontrajectory, ptstart, ptend)
    is_back = 1 - 1 * is_target_forward(motiontrajectory, ptstart, ptend)
    err_ballistic += is_back  # penalty for wrong direction
    
    # Normalized motion phase
    phase = normalized_motion_phase(screentrajectory, ptstart, ptend)

    # Correction onset is the point 
    # when no-disturbance error exceeds A*sigma and
    # the real error is decreasing
    ionsets = []
    for i in range(disturbanceindex, len(err_screen)-3):
        err_base_mean, err_base_covar =  baseline_llr.regress(phase[i])
        err_base_var = np.sqrt(err_base_covar)
        if (np.abs(err_ballistic[i]) > 1 * err_base_var) and \
            (np.abs(err_screen[i]) > np.abs(err_screen[i+1])) and \
            (np.abs(err_screen[i+1]) > np.abs(err_screen[i+2])) and \
            (np.abs(err_screen[i+2]) > np.abs(err_screen[i+3])):
            #(np.abs(err_ballistic[i]) > np.abs(err_screen[i])):
            ionsets.append(i)

    # Plot the detected onsets
    make_plots = True
    if make_plots:
        plt.plot(screentrajectory[:, 0], screentrajectory[:, 1])
        plt.plot(motiontrajectory[:, 0], motiontrajectory[:, 1])
        sp = ptstart
        plt.plot(sp[0], sp[1], marker='o', markersize=3,)
        gp = ptend
        plt.plot(gp[0], gp[1], marker='o', markersize=3,)
        
        plt.scatter(screentrajectory[:-1, 0], screentrajectory[:-1, 1], marker='o', s=15, 
                linewidths=4, c=err_screen, alpha=0.5, cmap=plt.cm.coolwarm)

        onsets = screentrajectory[ionsets]
        plt.scatter(onsets[:, 0], onsets[:, 1], marker='*', s=10, linewidths=4,)
        plt.axis("equal")
        if plotfilename is not None:
            plt.savefig(plotfilename)
            plt.close()
        else:
            plt.show()
        

    return onsets




if __name__ == "__main__":
    
    import analysis.delayedfeedback.datalayer as dl
    import analysis.delayedfeedback.database as db

    di = dl.DataInterface()
    di.data_type = dl.TrajectoryDataType.ScreenProjected
    
    subjectname = "Dominik"
    trials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == su.DisturbanceMode.NoDisturbance) \
            .filter(db.Trial.feedback_delay == -0.0) \
            .all()
    print("Number of trials: {}".format(len(trials)))


    motiontrajectories = [di.get_trial_opto_data(trial) for trial in trials]
    ptstarts = [x[0] for x in motiontrajectories]
    ptends = [trial.params["goal_position"] for trial in trials]
    errs, phases = compute_path_error_mean_var(motiontrajectories, ptstarts, ptends)
    valid_phases, valid_errs = select_valid_parts(errs, phases)
    llr = LocallyLinearRegression(np.concatenate(valid_phases), np.concatenate(valid_errs), tau=0.05)
    plot_regressed(llr)
    tp = TargetPredictor(motiontrajectories, ptstarts, ptends)
    for motiontrajectory, ptend, i in zip(motiontrajectories, ptends, range(len(ptends))):
        plotfilename = "./plots/predicted_target_{}.pdf".format(i)
        #tp.predict_target(motiontrajectory, ptend, plotfilename=None)

    #exit()

    trials = di.dbsession.query(db.Trial) \
            .join(db.Block, db.Session, db.Subject) \
            .filter(db.Subject.name == subjectname) \
            .filter(db.Trial.disturbance_mode == su.DisturbanceMode.NoDisturbance) \
            .filter(db.Trial.feedback_delay == -0.0) \
            .all()
    
    for trial in trials[0:100]:
        data = di.get_trial_opto_data(trial)
        t = di.get_trial_time_data(trial)
        print(trial.params)
        print("Block {}, trial {}".format(trial.block.number, trial.number))
        start_position = data[0]
        is_ballistic = is_ballistic_movement(llr, data, start_position, trial.params["goal_position"])
        print("is_ballistic", is_ballistic)
        plotfilename = "./plots/predicted_target_{}.pdf".format(i)
        tp.target_probability(data, trial.params["goal_position"])
        #tp.predict_target(data, trial.params["goal_position"], plotfilename=None)
        
        #onsets = detect_correction_onset(llr, data, t, 
        #        start_position, 
        #        trial.params["goal_position"], 
        #        trial.params["disturbance_mode"], 
        #        trial.params["disturbance_value"],
        #        disturbanceonsettime=None,
        #        plotfilename="./plots/threshold_correction_onset_{}.pdf".format(trial.number))
    exit()


    for trial in trials[0:100]:
        data = di.get_trial_opto_data(trial)
        t = di.get_trial_time_data(trial)
        print(trial.params)
        print("Block {}, trial {}".format(trial.block.number, trial.number))
        start_position = data[0]
        datavieved, err, dist_time, dist_index = compute_trial_error(data, t, 
                start_position, 
                trial.params["goal_position"], 
                trial.params["disturbance_mode"], 
                trial.params["disturbance_value"],
                disturbanceonsettime=None)
        
        plt.plot(datavieved[:, 0], datavieved[:, 1])
        plt.plot(data[:, 0], data[:, 1])
        sp = trial.params["start_position"]
        plt.plot(sp[0], sp[1], marker='o', markersize=3,)
        gp = trial.params["goal_position"]
        plt.plot(gp[0], gp[1], marker='o', markersize=3,)
        #plt.plot(ptenddisturbed[0], ptenddisturbed[1], marker='+', markersize=3)
        #plt.plot(pttrigger[0], pttrigger[1], marker='x', markersize=10)
        
        plt.scatter(datavieved[:-1, 0], datavieved[:-1, 1], marker='o', s=15, 
                linewidths=4, c=err, alpha=0.5, cmap=plt.cm.coolwarm)
        plt.axis("equal")
        plt.show()

        dist = normalized_motion_phase(data, 
                data[0], 
                trial.params["goal_position"])
        plt.plot(#dist[:-1], 
                err)
        plt.show()
    

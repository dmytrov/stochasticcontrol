import numpy as np
import matplotlib.pyplot as plt
import states.utils as su
import analysis.delayedfeedback.regression as re
import analysis.delayedfeedback.patherrors as pe
import analysis.delayedfeedback.datalayer as dl


def is_target_forward(motiontrajectory, ptstart, ptend):
    """ Computes ray-target forward sign.
        forward:
            positive for forward target (decreasing distance),
            negative for back target (increasing distance).
    """
    assert len(motiontrajectory.shape) == 2
    assert motiontrajectory.shape[1] == 2
    
    mv = np.diff(motiontrajectory, axis=0)  # motion vector
    vgoal = (ptend - motiontrajectory)  # target vector, unnormatized
    forward = np.sign(np.sum(mv * vgoal[:-1, :], axis=1))

    return forward


def normalized_dist_to_target(motiontrajectory, ptstart, ptend):
    d = np.linalg.norm(motiontrajectory-ptend, axis=1) / np.linalg.norm(ptstart - ptend)
    return d


def normalized_motion_phase(motiontrajectory, ptstart, ptend):
    tostart = np.linalg.norm(motiontrajectory-ptstart, axis=1)
    toend = np.linalg.norm(motiontrajectory-ptend, axis=1)
    d = tostart / (tostart + toend)
    return d


def get_start_point(motiontrajectory):
    # Fist trajectory point must be close to the home point
    i = np.argmax(np.logical_not(np.isnan(motiontrajectory)))
    return motiontrajectory[i]


def started_at_home(motiontrajectory, pthome, pttarget, phasethreshold=0.1):
    # Fist trajectory point must be close to the home point
    x0 = get_start_point(motiontrajectory)
    x0home = np.linalg.norm(x0 - pthome)
    x0target = np.linalg.norm(x0 - pttarget)
    phase = x0home / (x0home + x0target)
    correct = phase < phasethreshold
    
    if False:
        print("correct: ", correct)
        print("Start phase: {}".format(phase))
        print(pthome)
        plt.plot(motiontrajectory[:, 0], motiontrajectory[:, 1])
        plt.plot(pthome[0], pthome[1], "o")
        plt.plot(pttarget[0], pttarget[1], "*")
        plt.plot(x0[0], x0[1], "x")
        plt.axis("equal")
        plt.show()
    
    return correct


def target_reached(motiontrajectory, pthome, pttarget, phasethreshold=0.9):
    # Closest trajectory point to the disturbance corrected target must be close enough
    i = np.argmin(np.linalg.norm(motiontrajectory-pttarget, axis=-1))
    x1 = motiontrajectory[i]
    x1home = np.linalg.norm(x1 - pthome)
    x1target = np.linalg.norm(x1 - pttarget)
    phase = x1home / (x1home + x1target)
    correct = phase > phasethreshold

    if False:
        print("pttarget: ", pttarget)
        print("correct: ", correct)
        print("Start phase: {}".format(phase))
        print(pthome)
        plt.plot(motiontrajectory[:, 0], motiontrajectory[:, 1])
        plt.plot(pthome[0], pthome[1], "o")
        plt.plot(pttarget[0], pttarget[1], "*")
        plt.plot(x1[0], x1[1], "x")
        plt.axis("equal")
        plt.show()

    return correct


def path_data_present(motiontrajectory, pthome, pttarget, threshold=0.9):
    c = np.count_nonzero(np.logical_not(np.any(np.isnan(motiontrajectory), axis=-1)))
    r =  c / len(motiontrajectory)
    return r >= threshold



def is_ballistic_movement(baseline_llr, motiontrajectory, 
        ptstart, pttarget, 
        ballisticphasestart=0.1, 
        ballisticphaseend=0.2):
    
    phase = normalized_motion_phase(motiontrajectory, ptstart, pttarget)
    
    # In [phasestart, phaseend] interval error should not exceed the threshold
    err_ballistic = pe.raydistance_error(motiontrajectory, ptstart, pttarget)
    
    istart = np.argmax(phase > ballisticphasestart)
    iend =  np.argmax(phase > ballisticphaseend)

    if baseline_llr is None:
        # The trial is a baseline trial. Compare to a straight line
        return np.all(np.abs(err_ballistic[istart:iend]) < 0.5*np.linalg.norm(ptstart-pttarget))
        
    else:
        for i in range(istart, iend):
            m, c = baseline_llr(phase[i])
            v = np.sqrt(c)
            r = np.abs(m - err_ballistic[i]) / v
            if r > 3:
                return False
    return True


def acceleration(motiontrajectory):
    return np.diff(motiontrajectory, n=2, axis=0)


def abs_acceleration(motiontrajectory):
    return np.linalg.norm(acceleration(motiontrajectory), axis=-1)


def jerk(motiontrajectory):
    return np.diff(motiontrajectory, n=3, axis=0)


def abs_jerk(motiontrajectory):
    return np.linalg.norm(jerk(motiontrajectory), axis=-1)


def high_freq_energy(motiontrajectory, n=5):
    motiontrajectory = np.diff(motiontrajectory, axis=0)
    motiontrajectory = motiontrajectory - np.nanmean(motiontrajectory, axis=0)
    fft = np.fft.rfft(motiontrajectory, axis=0)
    spectrum = np.sum(np.abs(fft), axis=-1)
    sp = [np.sum(spectrum[i * int(len(spectrum)/n) : (i+1) * int(len(spectrum)/n)]**2) for i in range(n)]
    sp = sp / np.sum(sp)
    #print(sp)
    #plt.plot(motiontrajectory)
    #plt.plot(20*motiontrajectory)
    #plt.show()
    return sp[-1] 


def trial_is_valid(baseline_llr, motiontrajectory, pthome, pttarget, pttargetcorrected, realtime3D):
    ptstart = get_start_point(motiontrajectory)
    v1 = started_at_home(motiontrajectory, pthome, pttarget)
    v2 = target_reached(motiontrajectory, pthome, pttargetcorrected, phasethreshold=0.9)
    v3 = is_ballistic_movement(baseline_llr, motiontrajectory, ptstart, pttarget)
    v4 = path_data_present(motiontrajectory, pthome, pttarget)
    v5 = np.nanmax(abs_acceleration(motiontrajectory)) < 0.01
    v6 = np.nanmax(abs_jerk(motiontrajectory)) < 0.02
    v7 = high_freq_energy(realtime3D) < 0.005
    
    #acc = abs_acceleration(motiontrajectory)
    #jerk = abs_jerk(motiontrajectory)
    #print(np.nanmax(acc), np.nanmax(jerk))
    #if not v5 or not v6:
    #    plt.plot(jerk)
    #    plt.plot(acc)
    #    plt.plot(motiontrajectory)
    #    plt.show()

    valid = v1 and v2 and v3 and v4 and v5 and v6 and v7
    #if not valid:
    #    print("Invalid trial found")
    #    print(v1, v2, v3, v4, v5, v6, v7)
    return valid



def get_corrected_ptstart_ptend(
        ptstart, ptend, 
        disturbancemode, 
        disturbancevalue):
    """ Return disturbance corrected ptend.
        ptstart, ptend : [2] - vectors of exact home and target positions
    """
    if disturbancemode == su.DisturbanceMode.NoDisturbance:
        return ptstart, ptend

    # Rotation or translation disturbance inverse transformation
    mtrans = None
    if disturbancemode == su.DisturbanceMode.Rotation:
        mtrans = su.from_rotation_around(-disturbancevalue, ptstart)
    elif disturbancemode == su.DisturbanceMode.Translation:
        mtrans = su.from_translation(-np.array(disturbancevalue))
    ptstart_corrected = su.apply_disturbance(mtrans, np.array([ptstart]))[0]
    ptend_corrected = su.apply_disturbance(mtrans, np.array([ptend]))[0]
    return ptstart_corrected, ptend_corrected


def interpolate_missing_data(trajectory, tau=0.05):
    if not np.any(np.isnan(trajectory)):
        return trajectory

    xstars = np.linspace(0, 1, len(trajectory))
    ismissing = np.any(np.isnan(trajectory), axis=-1)
    llr = re.LocallyLinearRegression(xstars, trajectory, tau=tau)
    ystars = np.array([trajectory[i] if not ismissing[i] else llr.regress(xstars[i])[0] \
            for i in range(len(trajectory))])
    
    #plt.plot(xstars, trajectory[:, 0])
    #plt.plot(xstars, trajectory[:, 1])
    #plt.plot(xstars, ystars[:, 0], ".")
    #plt.plot(xstars, ystars[:, 1], ".")
    #plt.title("Interpolated missing data")
    #plt.show()
    return ystars
    


def resample_trajectory_normalize_path(trajectory, n):
    """ Resamples trajectory
            x : [N, D], D-dimensinoal trajectory
            n : int, number of resampling points
        Returns:
            [n, D], resampled path normalized trajectrory
    """
    dx = np.diff(trajectory, axis=0)
    dxnorm = np.linalg.norm(dx, axis=-1)
    path = np.hstack([[0], np.cumsum(dxnorm)])
    pathnormalized = path / path[-1]

    xstars = np.linspace(0, 1, n)
    #print(pathnormalized.shape, trajectory.shape)
    llr = LocallyLinearRegression(pathnormalized, trajectory)
    ystars = np.array([llr.regress(xstar)[0] for xstar in xstars])
    return ystars


def get_valid_indexes(data, phase, phasemin=0.05, phasemax=0.95):
    valid_start = np.argmax(phase[:-1] >= phasemin)
    valid_end = valid_start + np.argmax(phase[valid_start:] > phasemax)
    if valid_start == valid_end:
        valid_end = len(data)
    return valid_start, valid_end


class Trial(object):
    """ Database trial with additional methods and properties
    """
    def __init__(self, dbinterface, dbtrial, baselinellr=None):
        #print(dbtrial.params)
        self.dbtrial = dbtrial
        self.baselinellr = baselinellr
        self.motiontrajectory = dbinterface.get_trial_opto_data(dbtrial)
        self.realtime3D = dbinterface.get_trial_opto_data(dbtrial, data_type=dl.TrajectoryDataType.Recovered)
        self.pthome = tuple(dbtrial.params["start_position"])
        self.ptstart = tuple(get_start_point(self.motiontrajectory))
        self.feedbackdelay = dbtrial.params["feedback_delay"]
        self.disturbancemode = dbtrial.params["disturbance_mode"]
        self.disturbancevalue = dbtrial.params["disturbance_value"]
        if isinstance(self.disturbancevalue, list):
            self.disturbancevalue = tuple(self.disturbancevalue)
        self.pttarget = tuple(dbtrial.params["goal_position"])
        self.pthomecorrected, self.pttargetcorrected = get_corrected_ptstart_ptend(
                self.pthome, self.pttarget, 
                self.disturbancemode, self.disturbancevalue)
        self.ptstartcorrected, _ = get_corrected_ptstart_ptend(
                self.ptstart, self.pttarget, 
                self.disturbancemode, self.disturbancevalue)
        self.isvalid = trial_is_valid(baselinellr, self.motiontrajectory, 
                self.pthome, self.pttarget, self.pttargetcorrected, self.realtime3D)
        self.disturbed = not np.allclose(self.pttargetcorrected, self.pttarget)
        
        self.motiontrajectoryinterpolated = interpolate_missing_data(self.motiontrajectory)
        self.phase = normalized_motion_phase(self.motiontrajectoryinterpolated, 
                self.ptstart, self.pttarget)
        self.phasecorrected = normalized_motion_phase(self.motiontrajectoryinterpolated, 
                self.ptstart, self.pttargetcorrected)

        # Valid data start and end
        self.a, self.b = get_valid_indexes(self.motiontrajectoryinterpolated, self.phasecorrected, phasemin=0.05, phasemax=0.90)
        
        
        




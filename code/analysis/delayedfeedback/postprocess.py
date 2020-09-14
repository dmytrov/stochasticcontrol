import os
import sys
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import optotrak.calibrationcore as cc
from utils.logger import *
#import analysis.delayedfeedback.database as db
#import analysis.delayedfeedback.datalayer as dtl
import analysis.delayedfeedback.analyze_delayedfeedback as adf



datapath = "../../../data/delayedfeedback"
procpath = "../../../processed/delayedfeedback"



def read_optotrak_offline_trajectory(blockpath):
    opto_filename = os.path.join(os.path.dirname(__file__), blockpath, "REC-001.OPTO.npy")
    data = np.load(opto_filename).astype(float)
    data[np.abs(data) > 1.0e6] = np.NAN
    return data


def read_optotrak_odau(blockpath):
    odau_filename = os.path.join(os.path.dirname(__file__), blockpath, "REC-001.ODAU.npy")
    data = np.load(odau_filename).astype(float)
    return data
    

def read_online_trajectory(blockpath):
    nplogfilename = os.path.join(os.path.dirname(__file__), blockpath + "/datarecorder.pkl")
    print("Reading {}".format(nplogfilename))
    nplog = NPLog.from_file(nplogfilename)
    #print("Arrays logged: {}".format(nplog.get_names()))
    t, data = nplog.stack("realtimedata")
    framenumbers = np.squeeze(np.array([record.framenr for record in data]))

    trajectory = np.squeeze(np.array([record.data for record in data]))
    trajectory[np.abs(trajectory) > 1.0e6] = np.NAN

    x = np.arange(framenumbers[0], framenumbers[-1], 1)
    tr =  np.stack([np.interp(x, framenumbers, trajectory[:, 0]),
                    np.interp(x, framenumbers, trajectory[:, 1]),
                    np.interp(x, framenumbers, trajectory[:, 2]),], axis=-1)
    t = np.interp(x, framenumbers, t)

    frametimes = np.squeeze(np.array([record.timestamp for record in data]))
    frametimes = np.interp(x, framenumbers, frametimes)    
    
    # Best linear fit of frametimes to remove jitter
    n = len(frametimes)
    r = np.vstack([range(n), np.ones(n)])
    ab = frametimes.dot(np.linalg.pinv(r))
    tfit = ab.dot(r)

    return tfit, tr
        
    
def realign(data1, data2):
    n = 10000
    
    sh = 0
    r = np.ones([n, 4])
    r[:, :3] = data1[sh:sh+n]
    
    sh = 0
    d = np.ones([n, 4])
    d[:, :3] = data2[sh:sh+n]

    x = np.dot(r.T, np.linalg.pinv(d.T))
    
    r = r[:, :3]
    d = np.dot(x, d.T).T[:, :3]
    
    d = np.ones([data2.shape[0], 4])
    d[:, :3] = data2
    data2 = np.dot(x, d.T).T[:, :3]

    return data2


def list_dirs(dirpath):
    return [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,o))]
    
def list_dirs_tree(dirpath):
    return [x[0] for x in os.walk(dirpath)][1:]


def ensure_dir_exists(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)



def iterate_all_blocks(callback):
    subjectpathes = list_dirs(datapath)
    for subjectpath in subjectpathes:
        print("Session path \"{}\"".format(subjectpath))
        blockpaths = list_dirs(subjectpath)
        for blockpath in blockpaths:
            print("Block path \"{}\"".format(blockpath))
            callback(blockpath)



def recover_dropped_frames(blockpath):
    savepath = blockpath.replace(datapath, procpath)
    savepath = os.path.join(os.path.dirname(__file__), savepath)
    if not os.path.exists(os.path.join(savepath, "optical.npy")):
        print("Reading block {}".format(blockpath))
        exit()
        t, data = read_online_trajectory(blockpath)
        print("Recovered shape: {}".format(data.shape))

        print("Saving to: {}".format(savepath))
        ensure_dir_exists(savepath)
        np.save(os.path.join(savepath, "timestamps.npy"), t)
        np.save(os.path.join(savepath, "optical.npy"), data)


def project_to_screen(blockpath):
    recover_dropped_frames(blockpath)
    savepath = blockpath.replace(datapath, procpath)
    savepath = os.path.join(os.path.dirname(__file__), savepath)
    if not os.path.exists(os.path.join(savepath, "screen_trajectory.npy")):
        calibration = cc.ProjectionCalibration()
        calibration.load_from_file(os.path.join(datapath, "projection_calibration.pkl"))
        trajectory = np.load(os.path.join(savepath, "optical.npy"))
        screen_data = calibration.apply_calibration(trajectory)
        np.save(os.path.join(savepath, "screen_trajectory.npy"), screen_data)
        

def read_restored_trajectory(blockpath):
    recover_dropped_frames(blockpath)
    #print("Reading OPTO block {}".format(blockpath))
    savepath = blockpath.replace(datapath, procpath)
    savepath = os.path.join(os.path.dirname(__file__), savepath)
    timestamps = np.load(os.path.join(savepath, "timestamps.npy"))
    trajectory = np.load(os.path.join(savepath, "optical.npy"))
    return timestamps, trajectory


def read_projected_to_screen(blockpath):
    project_to_screen(blockpath)
    #print("Reading projected to screen data block {}".format(blockpath))
    savepath = blockpath.replace(datapath, procpath)
    savepath = os.path.join(os.path.dirname(__file__), savepath)
    timestamps = np.load(os.path.join(savepath, "timestamps.npy"))
    trajectory = np.load(os.path.join(savepath, "screen_trajectory.npy"))
    return timestamps, trajectory


def read_sync_EMG(blockpath):
    #print("Reading ODAU block {}".format(blockpath))
    data = read_optotrak_odau(blockpath)
    #print(data.shape)
    sync = data[:, 0]
    emg = data[:, 1:]
    return sync, emg


def filter_EMG(emg, fnyq=0.5 * 1200, fcut=15):
    b, a = signal.butter(2, fcut*1.25/fnyq)
    y = signal.filtfilt(b, a, np.abs(emg), axis=0, padlen=150)
    return y


def filter_trajectory(x, fnyq=0.5 * 120, fcut=15):
    
    b, a = signal.butter(2, fcut*1.25/fnyq)
    y = signal.filtfilt(b, a, x, axis=0, padlen=150)
    return y


def plot_EMG(emg, filtered):
    plt.plot(emg)
    plt.plot(filtered)
    plt.plot(np.arange(len(filtered))[::10], filtered[::10])
    plt.show()
    return

def plot_EMG_FFT(x):
    NFFT = 1024  # the length of the windowing segments
    Fs = 1200 #int(1.0 / dt)  # the sampling frequency

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(np.abs(x))
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the matplotlib.image.AxesImage instance representing the data in the plot
    plt.show()

    #np.save(os.path.join(savepath, "EMG.npy"), data)


def block_processor_filter_emg(blockpath):
    sync, emg = read_sync_EMG(blockpath)
    #emg = emg[:20000]
    filtered = filter_EMG(emg, fcut=1)[::10]  # every 10-th
    t, trajectory = read_restored_trajectory(blockpath)
    print(filtered.shape, trajectory.shape)
    velocity = np.diff(trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)
    n = len(acceleration)
    print(n, filtered.shape, trajectory.shape)
    if np.abs(len(filtered) - len(trajectory)) > 5:
        return
    nx = 4
    lag = 45
    xy = np.hstack([filtered[:n-lag], trajectory[lag:n], velocity[lag:n], acceleration[lag:n]])
    #xy = np.hstack([filtered[:n], trajectory[:n], velocity[:n], acceleration[:n]])


    # Remove NaNs
    invalid = np.any(np.isnan(xy), axis=1)
    xy = xy[np.logical_not(invalid)]
    
    # Center the data
    xy_mean = np.nanmean(xy, axis=0)
    cxy = xy - xy_mean
    
    # Whiten the data
    w = 1.0 / np.sqrt(np.sum(cxy**2, axis=0)/len(cxy))
    wxy = w * cxy

    # Cross-correlation
    a, b = wxy[:, :nx], wxy[:, nx:]
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            cc = np.abs(signal.correlate(a[:, i], b[:, j]))
            t = len(cc)
            t0 = int(t/2)
            lag = np.argmax(cc)-t0
            print(i, j, lag)
    plt.plot(range(-t0, t-t0), cc)
    plt.show()

    # Covariance matrix
    wxytwxy = np.dot(wxy.T, wxy)/len(wxy)
    plt.imshow(wxytwxy)
    plt.show()
    
    # Decode trajectory from EMG
    cov_xy = wxytwxy[:nx, nx:]
    cov_yx = wxytwxy[nx:, :nx]
    cov_xx = wxytwxy[:nx, :nx]
    print(wxytwxy.shape, cov_xy.shape)
    ystar = np.dot(cov_yx, np.linalg.inv(cov_xx)).dot(wxy[:, :nx].T).T
    print(ystar.shape)

    n = 2000
    plt.plot(ystar[:n, :3] + [0, 5, 10])
    plt.plot(wxy[:n, nx:nx+3] + [0, 5, 10])
    plt.show()

    plt.plot(wxy[:, nx+1], wxy[:, nx+2])
    plt.plot(ystar[:, 1], ystar[:, 2])
    plt.show()
    #exit()

    #plot_EMG(emg, 10*filtered)
    #exit()


def block_processor_optmial_filter(blockpath):
    sync, emg = read_sync_EMG(blockpath)
    nemg = emg.shape[-1]
    
    t, trajectory = read_restored_trajectory(blockpath)
    velocity = np.diff(trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)

    #velocity = filter_trajectory(velocity, fcut=30)  # returns all NaNs
    #acceleration = filter_trajectory(acceleration)  # returns all NaNs
    
    
    filtertype = "blocksum"
    if filtertype == "blocksum":
        # Block sum filer
        emg = np.abs(emg)
        emg = np.reshape(emg, newshape=(len(trajectory), emg.shape[-1], -1))
        emg = np.sum(emg, axis=-1)
    elif filtertype == "lowpass":
        emg = filter_EMG(emg, fcut=50)[::10]  # every 10-th
    
    #plt.plot(emg[:2000])
    #plt.show()


    # Low-pass filter
    
    n = len(acceleration)
    if np.abs(len(emg) - len(trajectory)) > 5:
        return

    minlag = -60
    maxlag = -10
    
    x = np.vstack([emg[-minlag+lag:n+lag, i] for i in range(emg.shape[-1]) for lag in range(minlag, maxlag)]).T
    #x = np.hstack([emg[-minlag+lag:n+lag] for lag in range(minlag, maxlag)])
    y = np.hstack([trajectory[-minlag:n], velocity[-minlag:n], acceleration[-minlag:n]])
    xy = np.hstack([x, y])
    
    # Remove NaNs
    invalid = np.any(np.isnan(xy), axis=1)
    ninvalid = np.count_nonzero(invalid)
    print("Invalid samples: {}% ({} of {})".format(100*ninvalid/len(xy), ninvalid, len(xy)))
    xy = xy[np.logical_not(invalid)]
    
    # Center the data
    xy_mean = np.nanmean(xy, axis=0)
    cxy = xy - xy_mean
    
    # Whiten the data
    w = 1.0 / np.sqrt(np.sum(cxy**2, axis=0)/len(cxy))
    wxy = w * cxy

    nx = x.shape[1]
    print("Regressor size: {}".format(nx))
    
    # Covariance matrix
    wxytwxy = np.dot(wxy.T, wxy)/len(wxy)
    #plt.imshow(wxytwxy)
    #plt.title("Covariance matrix")
    #plt.show()

    
    # Decode trajectory from EMG
    cov_xy = wxytwxy[:nx, nx:]
    cov_yx = wxytwxy[nx:, :nx]
    cov_xx = wxytwxy[:nx, :nx]
    cov_yy = wxytwxy[nx:, nx:]
    
    # PCA
    E, V = np.linalg.eigh(cxy.T.dot(cxy)[:4, :4]/len(cxy))
    print("EMG eigenvalues: {}".format(E/np.max(E)))
    
    # Optimal decoder
    decoder = np.dot(cov_yx, np.linalg.inv(cov_xx))
    decoder_covar = cov_yy - np.dot(cov_yx, np.linalg.inv(cov_xx).dot(cov_xy))
    #print(decoder.shape)
    #plt.imshow(decoder)
    #plt.show()
    
    for i in range(0, 3):
        for iemg in range(nemg):
            plt.subplot(3, nemg, iemg + nemg*i+1)
            plt.plot(decoder[i*3:(i+1)*3, iemg*nx/nemg:(iemg+1)*nx/nemg].T)
            plt.title("Ch {}, {}".format(iemg, ["position", "velocity", "acceleration"][i]))
    plt.show()
    
    ystar = decoder.dot(wxy[:, :nx].T).T
    
    n = 20000
    for f, feat in zip(range(0, 3), ["position", "velocity", "acceleration"]):
        fig = plt.figure()
        fig.suptitle(feat)
        for i, coorname in zip(range(0, 3), ["x", "y", "z"]):  # coordinates
            plt.subplot(3, 1, i+1) 
            k = f*3 + i
            yi = ystar[:n, k]
            ei = decoder_covar[k, k]
            plt.fill_between(range(len(yi)), yi-ei, yi+ei, alpha=0.2, linewidth=1)
            plt.plot(yi, linewidth=1)
            plt.plot(wxy[:n, nx+k], linewidth=1)
            plt.title(coorname)

        plt.show()


if __name__ == "__main__":
    # Recover the missing frames
    iterate_all_blocks(recover_dropped_frames)
    iterate_all_blocks(project_to_screen)
    #exit()


    # Filter the EMG
    iterate_all_blocks(block_processor_optmial_filter)
    exit()
    
    
    datapath = "../../../data/delayedfeedback"
    procpath = "../../../processed/delayedfeedback"

    subjectpathes = list_dirs(datapath)
    for subjectpath in subjectpathes:
        print("Session path \"{}\"".format(subjectpath))
        blockpaths = list_dirs(subjectpath)
        for blockpath in blockpaths:
            print("Block path \"{}\"".format(blockpath))

            savepath = blockpath.replace(datapath, procpath)
            if not os.path.exists(savepath):
                print("Reading block {}".format(blockpath))
                t, data = read_online_trajectory(blockpath)
                print("Recovered shape: {}".format(data.shape))
    
                print("Saving to: {}".format(savepath))
                ensure_dir_exists(savepath)
                np.save(os.path.join(savepath, "timestamps.npy"), t)
                np.save(os.path.join(savepath, "optical.npy"), data)
            
    
    

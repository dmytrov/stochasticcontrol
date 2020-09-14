import numpy as np
import linalg.routines as lr



def map_calibrated_projection(env, markers):
    return env.calib_trans.apply_calibration(markers)



def map_calibrated_range(env, markers):
    return env.calib_trans.apply_calibration(markers)



def map_to_angles(markers):
    markers = np.array(markers)
    x = lr.lines_angle(markers[0], markers[1], markers[2], markers[3])
    y = lr.lines_angle(markers[4], markers[5], markers[6], markers[7])
    return np.array([[x, y]])



def map_to_angles_from_mouse_emulation(markers):
    x = markers[0][0]
    y = markers[0][1]
    return np.array([[x, y]])




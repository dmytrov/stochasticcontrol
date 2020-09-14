""" No PyGame dependency
"""

import numpy as np


class DisturbanceMode(object):
    NoDisturbance = 0
    Translation = 1  # translation vector
    Rotation = 2  # rotation around start position


def from_translation(translation):
    mtrans = np.identity(3)
    mtrans[0:2, 2] = translation
    return mtrans


def from_rotation(angle):
    mtrans = np.identity(3)
    mtrans[0:2, 0:2] = [[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]]
    return mtrans


def from_rotation_around(angle, center):
    mt = np.identity(3)
    mt[0:2, 2] = center
    
    mr = np.identity(3)
    mr[0:2, 0:2] = [[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]]

    mtrans = mt.dot(mr).dot(np.linalg.inv(mt))
    return mtrans


def apply_disturbance(mtrans, markerposition):
    markerposition3 = np.hstack([markerposition, np.ones(shape=[markerposition.shape[0], 1])])
    return mtrans.dot(markerposition3.T).T[:, :2]
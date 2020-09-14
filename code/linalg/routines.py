import numpy as np


def lines_closest_pts(a, b, c, d):
    """
    Returns two closest pints on the lines AB and CD, 
    which detetermines the distance between the lines.
    a, b, c, d: [3] numpy vectors
    return: n, m: [3] numpy vectors of the closest points  
    """
    A = np.array([[(b-a).dot(b-a), -(b-a).dot(d-c)],
                  [(d-c).dot(b-a), -(d-c).dot(d-c)]])
    B = -np.array([(b-a).dot(a-c), (d-c).dot(a-c)]) 
    x = np.linalg.inv(A).dot(B)
    n = a + x[0] * (b-a)
    m = c + x[1] * (d-c)
    return n, m

def lines_angle(a, b, c, d):
    """
    Returns the angle between the lines AB and CD, 
    a, b, c, d: [3] numpy vectors
    return: alpha: [3] numpy vector  
    """
    A = (a-b) / np.linalg.norm(a-b)
    B = (c-d) / np.linalg.norm(c-d)
    alpha = np.arccos(np.dot(A, B))
    return alpha


if __name__ == "__main__":
    a = np.array([ 1, 2, -2])
    b = np.array([ 2, 3, -2])
    c = np.array([ 0,-1, 5])
    d = np.array([ 0, 1, 5])
    n, m = lines_closest_pts(a, b, c, d)
    assert np.allclose(n, np.array([ 0.,  1., -2.]))
    assert np.allclose(m, np.array([ 0.,  1.,  5.]))
    
    a = np.array([ 0, 0, -3])
    b = np.array([ 1, 1, -3])
    c = np.array([ 0, 0, 5])
    d = np.array([ 1, 0, 5])
    alpha = lines_angle(a, b, c, d)
    assert np.allclose(alpha, np.pi / 4)
    
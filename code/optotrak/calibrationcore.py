"""
This allows calibration (alignment) of Optotrak and ViewPixx w.r.t the setup position
"""
import pickle
import numpy as np



class Calibration(object):
    def __init__(self, features_func=None):
        self.features_func = features_func  # computes features from measurements (angles from points e.g.)
        self.defaultfilename = "calibration.pkl"
        

    def save_to_file(self, filename=None):
        raise NotImplementedError()


    def load_from_file(self, filename=None):
        raise NotImplementedError()


    def apply_calibration(self, measurements):
        raise NotImplementedError()



class ProjectionCalibration(Calibration):
    def __init__(self, win=None):
        super(ProjectionCalibration, self).__init__()
        self.defaultfilename = "projection_calibration.pkl"
        if win is not None: 
            self.screen_resolution_x = win.size[0]
            self.screen_resolution_y = win.size[1]
            if win.units == "pix":
                pass
            elif win.units == "height":
                self.screen_resolution_x = self.screen_resolution_x / self.screen_resolution_y
                self.screen_resolution_y = 1.0
            elif win.units == "norm":
                self.screen_resolution_x = 1.0
                self.screen_resolution_y = 1.0
        else:
            self.screen_resolution_x = 800
            self.screen_resolution_y = 600
        self.screen_resolution = np.array([self.screen_resolution_x, self.screen_resolution_y])

        gridcount = 3
        grid = np.stack(np.meshgrid(np.linspace(0, self.screen_resolution_x, gridcount),
                                         np.linspace(0, self.screen_resolution_y, gridcount)))
        screen_points = np.reshape(grid, [2, -1]).T

        gridscale = 0.9
        render_center = 0.5 * self.screen_resolution
        self.render_query_points = gridscale * (screen_points - render_center)
        self.M_calibration = None


    def save_to_file(self, filename=None):
        if filename is None:
            filename = self.defaultfilename
        with open(filename,"wb") as df:
            pickle.dump(self.M_calibration, df)


    def load_from_file(self, filename=None):
        try:
            if filename is None:
                filename = self.defaultfilename
            with open(filename,"rb") as df:
                self.M_calibration = pickle.load(df, encoding="latin1")
            return True
        except IOError as e:
            return False


    def apply_calibration(self, measurements):
        if self.features_func is not None:
            measurements = self.features_func(measurements)
        x = np.hstack([measurements, np.ones([len(measurements), 1])]).T
        res = self.M_calibration.dot(x).T[:, 0:2]
        return res
    

def fit_transformation(renderpoints, measurements):
    n = renderpoints.shape[0]
    assert np.array_equal(renderpoints.shape, [n, 2])
    assert np.array_equal(measurements.shape, [n, 3])
    x = np.hstack([measurements, np.ones([n, 1])]).T
    y = np.hstack([renderpoints, np.zeros([n, 1]), np.ones([n, 1])]).T
    # Solve Ax = y equation w.r.t. A
    A = np.linalg.inv(x.dot(x.T)).dot(x).dot(y.T)
    return A.T



def compute_optotrak_to_screen_transformation(renderpoints, measurements):
    """
    renderpoints, measurements - points on a plane 
    """
    n = renderpoints.shape[0]
    assert np.array_equal(renderpoints.shape, [n, 2])
    assert np.array_equal(measurements.shape, [n, 3])

    # Add artificial measurements to constrain parallel projection
    mc = measurements - np.mean(measurements, axis=0)
    s, v, d = np.linalg.svd(mc)
    normal = d[2]
    renderpoints = np.vstack([renderpoints, renderpoints])
    measurements = np.vstack([measurements, measurements + 100.0 * normal])

    return fit_transformation(renderpoints, measurements)



class NormalizedRangeCalibration(Calibration):
    def __init__(self, features_func=None):
        super(NormalizedRangeCalibration, self).__init__(features_func)
        self.defaultfilename = "normalized_calibration.pkl"
        self.v = None
        self.b = None
    

    def compute_calibration_range(self, minmeasurement, midmeasurement, maxmeasurement):
        if self.features_func is not None:
            minmeasurement = self.features_func(minmeasurement)
            midmeasurement = self.features_func(midmeasurement)
            maxmeasurement = self.features_func(maxmeasurement)
        dmin = np.array([np.linalg.norm(np.ravel(v)) for v in minmeasurement - midmeasurement])
        dmax = np.array([np.linalg.norm(np.ravel(v)) for v in maxmeasurement - midmeasurement])
        dmin = np.linalg.norm(minmeasurement - midmeasurement, axis = 0)
        dmax = np.linalg.norm(maxmeasurement - midmeasurement, axis = 0)
        damplitude = np.min(np.vstack([dmin, dmax]), axis=0)
        self.v = ((maxmeasurement - minmeasurement) / (dmin + dmax) / damplitude)
        self.b = -midmeasurement


    def apply_calibration(self, measurement):
        if self.features_func is not None:
            measurement = self.features_func(measurement)
        return self.v * (measurement + self.b)
        
    
    def save_to_file(self, filename=None):
        if filename is None:
            filename = self.defaultfilename
        with open(filename,"wb") as df:
            pickle.dump((self.v, self.b), df)


    def load_from_file(self, filename=None):
        try:
            if filename is None:
                filename = self.defaultfilename
            with open(filename,"rb") as df:
                sself.v, self.b = pickle.load(df)
            return True
        except IOError as e:
            return False



if __name__ == "__main__":
    minmeasurement = np.array([[0, 30], 
                               [0, 1],
                               [0, 1]])
    midmeasurement = np.array([[5, 20], 
                               [5, 0.5],
                               [0, 2]])
    maxmeasurement = np.array([[20, 0], 
                               [10, 0],
                               [0, 4]])
    cal = NormalizedRangeCalibration()
    cal.compute_calibration_range(minmeasurement, midmeasurement, maxmeasurement)
    print(cal.apply_calibration(minmeasurement))
    print(cal.apply_calibration(midmeasurement))
    print(cal.apply_calibration(maxmeasurement))

    minmeasurement = np.array([[-0.6, -0.4]])
    midmeasurement = np.array([[0, 0]])
    maxmeasurement = np.array([[0.6, 0.4]])
    cal = NormalizedRangeCalibration()
    cal.compute_calibration_range(minmeasurement, midmeasurement, maxmeasurement)
    print(cal.apply_calibration(minmeasurement))
    print(cal.apply_calibration(midmeasurement))
    print(cal.apply_calibration(maxmeasurement))


if __name__ == "__main_":
    cc = ProjectionCalibration()
    bias = np.array([0.0, 0.0, -2500.0])
    factor = 0.1 * np.array([1.0, -1.0, 1.0])
    A = np.array([[0.1, 0.0, 0.0, 1.0],
                  [0.0,-0.1, 0.0, 2.0],
                  [0.0, 0.0, 0.1, 3.0],
                  [0.0, 0.0, 0.0, 1.0]])
    Ainv = np.linalg.inv(A)
    n = cc.render_query_points.shape[0]
    y = np.hstack([cc.render_query_points, np.zeros([n, 1]), np.ones([n, 1])])
    measurements = Ainv.dot(y.T)
    measurements += 0.01 * np.random.normal(size=measurements.shape)
    measurements = measurements.T
    Astar = compute_optotrak_to_screen_transformation(cc.render_query_points, measurements[:, :-1])
    render_points = Astar.dot(measurements.T).T[:, 0:2] - cc.render_query_points

    print(Astar)
    print(render_points-cc.render_query_points)
    
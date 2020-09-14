import time
import gc
from threading import Thread
import wrapt
import numpy as np
import optotrak.ndiapiconstants
import optotrak.databuffer as db
from utils.realtimeclock import rtc
import logging
import utils.logger as ulog
import gp.equations as gpe


ndi = optotrak.ndiapiconstants.NDI


class OptotrakRealtimeData(object):
    """
    Single frame data storage
    """
    def __init__(self, framenr, data, timestamp):
        self.framenr = framenr
        self.data = data
        self.timestamp = timestamp

    def as_numpy(self):
        return np.array(self.data)


class TimestampedDataBuffer(db.DataBuffer):
    def __init__(self, capacity=1, dataclass=OptotrakRealtimeData):
        super(TimestampedDataBuffer, self).__init__(capacity, dataclass)

    @wrapt.synchronized
    def select_values_by_timestamp(self, t=0.0, halfnsamples=2):
        """
        Returns 2*halfnsamples
        ---------------------------------> t
         |        |    |     |        |
                       t              
        """
        #t = rtc() + dt  # requested data timestamp
        i = 0
        while i < len(self.buffer) and self.buffer[i].timestamp < t:
            i += 1
        i0 = max(0, i-halfnsamples)
        i1 = min(len(self.buffer), i+halfnsamples)
        res = self.buffer[i0:i1]
        return res


class OptotrakDataRecorder(object):
    def __init__(self, optotrak=None):
        self.optotrak = optotrak
        self.realtimedatabuffer = TimestampedDataBuffer(capacity=10, dataclass=OptotrakRealtimeData)
        self.pullingthread = None
        self.pullingthreadactiveflag = False
        self.nplog = None
        self._verbose = False
        self._sleep_period = None
        
    @property
    def sleep_period(self):
        return self._sleep_period

    @sleep_period.setter
    def sleep_period(self, value):
        self._sleep_period = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def init_optotrak(self, nplogfilename=None):
        if nplogfilename is None:
            nplogfilename = "datarecorder.pkl"
        self.nplog = ulog.setup_numpy_logger(__name__, nplogfilename=nplogfilename)
        
        try:
            res = self.optotrak.TransputerLoadSystem("system")
            if res != 0: raise IOError(res)
            time.sleep(1.0)

            res = self.optotrak.TransputerInitializeSystem(ndi.OPTO_LOG_ERRORS_FLAG)
            if res != 0: raise IOError(res)

            res = self.optotrak.OptotrakLoadCameraParameters("standard")
            if res != 0: raise IOError(res)

        except IOError as err:
            print("OptotrakDataRecorder: failed to init with error: {}".format(self.optotrak.OptotrakGetErrorString()))
            print("OptotrakDataRecorder: shutting down Optotrak after the error...")
            self.optotrak.OptotrakDeActivateMarkers()
            self.optotrak.TransputerShutdownSystem()
            raise


    def setup_optotrak_collection(self, nummarkers=4, collecttime=10.0, 
                        datafps=120, buffer_capacity=10,
                        stream_push_mode=0,
                        datafilename=None):
        """
        Optotrak collection must be setup last
        """
        if datafilename is None:
            datafilename = "REC#001.OPT" 
        self.realtimedatabuffer.capacity = buffer_capacity

        try:
            res = self.optotrak.OptotrakSetupCollection(
                nummarkers,         # Number of markers in the collection.
                float(datafps),     # Frequency to collect data frames at.
                2500.0,             # Marker frequency for marker maximum on-time.
                30,                 # Dynamic or Static Threshold value to use.
                160,                # Minimum gain code amplification to use.
                stream_push_mode,   # Stream mode for the data buffers.
                0.35,               # Marker Duty Cycle to use.
                7.0,                # Voltage to use when turning on markers.
                collecttime,        # Number of seconds of data to collect.
                0.0,                # Number of seconds to pre-trigger data by.
                ndi.OPTOTRAK_BUFFER_RAW_FLAG | ndi.OPTOTRAK_GET_NEXT_FRAME_FLAG)
            if res != 0: raise IOError(res)
            time.sleep(1.0)

            res = self.optotrak.OptotrakActivateMarkers()
            if res != 0: raise IOError(res)

            res = self.optotrak.DataBufferInitializeFile(ndi.OPTOTRAK, datafilename.encode("ascii", "ignore"))
            if res != 0: raise IOError(res)

        except IOError as err:
            print("OptotrakDataRecorder: failed to setup Optotrak collection with error: {}".format(self.optotrak.OptotrakGetErrorString()))
            print("OptotrakDataRecorder: shutting down Optotrak after the error...")
            self.optotrak.OptotrakDeActivateMarkers()
            self.optotrak.TransputerShutdownSystem()
            raise

    def setup_odau_collection(self, numchannels=4, collecttime=10.0,
                        odauID=ndi.ODAU1,
                        analog_gain=1, # divider for [-10, 10] Volts range
                        digital_port_mode=ndi.ODAU_DIGITAL_PORT_OFF,
                        collection_frequency=5000.0,
                        scan_frequqncy=90000.0,
                        stream_push_mode=0,
                        datafilename=None):
        """
        ODAU collection must be setup first
        """
        if datafilename is None:
            datafilename = "REC#001.ODAU" 
        
        try:
            res = self.optotrak.OdauSetupCollection(
                odauID,                         # Id the ODAU the parameters are for. 
                numchannels,                    # Number of analog channels to collect. 
                analog_gain,                    # Gain to use for the analog channels.
                digital_port_mode,              # Mode for the Digital I/O port.
                float(collection_frequency),    # Frequency to collect data frames at. 
                float(scan_frequqncy),          # Frequency to sample the channels at within one frame. 
                stream_push_mode,               # Stream mode for the data buffers. 
                float(collecttime),             # Number of seconds of data to collect. 
                0.0,                            # Number of seconds to pre-trigger data by. 
                0)      
            if res != 0: raise IOError(res)
            
            res = self.optotrak.DataBufferInitializeFile(odauID, datafilename.encode("ascii", "ignore"))
            if res != 0: raise IOError(res)

        except IOError as err:
            print("OptotrakDataRecorder: failed to setup ODAU collection with error: {}".format(self.optotrak.OptotrakGetErrorString()))
            print("OptotrakDataRecorder: shutting down Optotrak after the error...")
            self.optotrak.OptotrakDeActivateMarkers()
            self.optotrak.TransputerShutdownSystem()
            raise

    def _pulling_thread_function(self):
        try:
            gc.disable()
            self.pullingthreadactiveflag = True
            res = self.optotrak.DataBufferStart()
            if res != 0: raise Exception(res)
            res = self.optotrak.RequestLatest3D()
            if res != 0: raise Exception(res)

            callcounterDataBufferWriteData = 0
            uSpoolComplete = False
            t0 = time.time()
            prevFramenumber = 0
            while not uSpoolComplete:
                (res, uRealtimeDataReady, uSpoolComplete,
                    uSpoolStatus, nFrames) = self.optotrak.DataBufferWriteData()
                if res != 0: raise Exception(res) 
                callcounterDataBufferWriteData += 1  
                if uRealtimeDataReady:
                    timestamp = rtc()
                    (res, uFrameNumber, uElements, uFlags,
                        p3dData) = self.optotrak.DataReceiveLatest3D()
                    if res != 0: raise Exception(res)
                    if self.verbose:
                        print("Frame Number: ", uFrameNumber)
                        print("Elements    : ", uElements)
                        print("Flags       : ", uFlags)
                        print(p3dData)
                    framesdelta = uFrameNumber - prevFramenumber
                    prevFramenumber = uFrameNumber
                    if framesdelta != 1:
                        print("Frame: {}. SKIPPED FRAMES: {}".format(uFrameNumber, framesdelta-1))
                    realtimedata = OptotrakRealtimeData(uFrameNumber, p3dData, timestamp)
                    self.realtimedatabuffer.add(realtimedata)
                    self.nplog.info(ulog.NPRecord("realtimedata", realtimedata))
                    res = self.optotrak.RequestLatest3D()
                    if res != 0: raise Exception(res)
                    if not self.pullingthreadactiveflag:
                        res = self.optotrak.DataBufferStop()
                        if res != 0: raise Exception(res)
                    uRealtimeDataReady = 0
                if self._sleep_period is not None:
                    time.sleep(self._sleep_period)
            t1 = time.time()
            dt = t1 - t0
            print("OptotrakDataRecorder: pulling complete.")
            print("DataBufferWriteData: {} calls/sec".format(callcounterDataBufferWriteData/dt))
        except IOError as err:
            print("OptotrakDataRecorder: failed to pull with error: {}".format(self.optotrak.OptotrakGetErrorString()))
            raise

        finally:
            gc.enable()
            print("OptotrakDataRecorder: shutting down Optotrak...")
            self.optotrak.OptotrakDeActivateMarkers()
            self.optotrak.TransputerShutdownSystem()
            print("OptotrakDataRecorder: thread finished")

    def start_pulling_thread(self):
        self.pullingthread = Thread(target=self._pulling_thread_function)
        self.pullingthread.start()

    def stop_pulling_thread(self):
        if self.pullingthread is not None:
            self.pullingthreadactiveflag = False
            self.pullingthread.join()

    def waitfillbuffer(self):
        while self.realtimedatabuffer.is_empty() and self.pullingthread is not None and self.pullingthread.isAlive():
            time.sleep(0.1)
        return not self.realtimedatabuffer.is_empty()

    def select_values_by_timestamp(self, t=0.0, halfnsamples=3):
        if self.pullingthread is not None and self.pullingthread.isAlive():
            samples = self.realtimedatabuffer.select_values_by_timestamp(t, halfnsamples)
            x = [sample.timestamp for sample in samples]
            y = [sample.data for sample in samples]
            return (x, y)
        else:
            return (None, None)

    def get_value_by_timestamp(self, t, halfnsamples=3):
        x, y = self.select_values_by_timestamp(t, halfnsamples)
        if x is not None:
            return OptotrakDataRecorder.GP_prediction(x, y, t).tolist()
        else:
            return None

    def get_value_by_now_plus_dt(self, dt):
        return self.get_value_by_timestamp(rtc() + dt)

    def shutdown_on_disconnect(self):
        return True

    @classmethod
    def GP_prediction(cls, x, y, xstar, visibleonly=True):
        d = len(y[0])
        x = np.array([x]).T
        y = np.array([np.ravel(yi) for yi in y])
        if visibleonly:
            ibad = y < -1e20
            ma = np.ma.array(y, mask=ibad)
            ms = np.mean(ma, axis=0)
            y[ibad] = (np.ones_like(y) * ms[np.newaxis, :])[ibad]
            
        value = gpe.conditional(x, y, np.array([[xstar]]))
        if visibleonly:
            ibad = value < -1e20
            value[ibad] = np.nan
            j = np.ones_like(value) * ms[np.newaxis, :]
            value[j.mask] = np.nan
        value = np.reshape(value, [d, -1])
        return value


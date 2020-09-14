import time
from threading import Thread
import numpy as np
import optotrak.ndiapiconstants
import optotrak.databuffer as db

ndi = optotrak.ndiapiconstants.NDI

class OptotrakRealtimeData(object):
    """
    Single frame data storage
    """
    def __init__(self, framenr, data):
        self.framenr = framenr
        self.data = data

    def as_numpy(self):
        return np.array(self.data)


class OptotrakDataPuller(object):
    def __init__(self, optotrak):
        self.optotrak = optotrak
        self.realtimedatabuffer = db.DataBuffer(capacity=10, dataclass=OptotrakRealtimeData)
        self.pullingthread = None
        self.pullingthreadactiveflag = True
        self.print_messages = False
        self.sleep_period = None

    def init_optotrak(self, nummarkers=4, collecttime=10.0, datafps=120):
        try:
            res = self.optotrak.TransputerLoadSystem("system")
            if res != 0: raise IOError(res)
            time.sleep(1)

            res = self.optotrak.TransputerInitializeSystem(ndi.OPTO_LOG_ERRORS_FLAG)
            if res != 0: raise IOError(res)

            res = self.optotrak.OptotrakLoadCameraParameters("standard")
            if res != 0: raise IOError(res)

            res = self.optotrak.OptotrakSetupCollection(
                nummarkers,     # Number of markers in the collection.
                float(datafps), # Frequency to collect data frames at.
                2500.0,         # Marker frequency for marker maximum on-time.
                30,             # Dynamic or Static Threshold value to use.
                160,            # Minimum gain code amplification to use.
                1,              # Stream mode for the data buffers.
                0.35,           # Marker Duty Cycle to use.
                7.0,            # Voltage to use when turning on markers.
                collecttime,    # Number of seconds of data to collect.
                0.0,            # Number of seconds to pre-trigger data by.
                ndi.OPTOTRAK_BUFFER_RAW_FLAG | ndi.OPTOTRAK_GET_NEXT_FRAME_FLAG)
            if res != 0: raise IOError(res)
            time.sleep(1.0)

            res = self.optotrak.OptotrakActivateMarkers()
            if res != 0: raise IOError(res)

            res = self.optotrak.DataBufferInitializeFile( ndi.OPTOTRAK, "R#001.S05" )
            if res != 0: raise IOError(res)

        except IOError as err:
            print("Data puller: failed to init with error: {}".format(self.optotrak.OptotrakGetErrorString()))
            print("Data puller: shutting down Optotrak after the error...")
            self.optotrak.OptotrakDeActivateMarkers()
            self.optotrak.TransputerShutdownSystem()
            raise

    def _pulling_thread_function(self):
        try:
            res = self.optotrak.DataBufferStart()
            if res != 0: raise Exception(res)
            res = self.optotrak.RequestLatest3D()
            if res != 0: raise Exception(res)

            uSpoolComplete = False
            while not uSpoolComplete and self.pullingthreadactiveflag:
                (res, uRealtimeDataReady, uSpoolComplete,
                    uSpoolStatus, nFrames) = self.optotrak.DataBufferWriteData()
                if res != 0: raise Exception(res)   
                if uRealtimeDataReady:
                    (res, uFrameNumber, uElements, uFlags,
                        p3dData) = self.optotrak.DataReceiveLatest3D()
                    if res != 0: raise Exception(res)
                    if self.print_messages:
                        print("Frame Number: ", uFrameNumber)
                        print("Elements    : ", uElements)
                        print("Flags       : ", uFlags)
                        print(p3dData)
                    realtimedata = OptotrakRealtimeData(uFrameNumber, p3dData)
                    self.realtimedatabuffer.add(realtimedata)
                    res = self.optotrak.RequestLatest3D()
                    if res != 0: raise Exception(res)
                    uRealtimeDataReady = 0
                if self.sleep_period is not None:
                    time.sleep(self.sleep_period)
            #res = self.optotrak.OptotrakDeActivateMarkers()
            #if res != 0: raise Exception(res)
            print("Data puller: pulling complete.")
        except IOError as err:
            print("Data puller: failed to pull with error: {}".format(self.optotrak.OptotrakGetErrorString()))
            raise
        finally:
            print("Data puller: shutting down Optotrak...")
            self.optotrak.OptotrakDeActivateMarkers()
            self.optotrak.TransputerShutdownSystem()
            print("Data puller: thread finished")

    def start_pulling_thread(self):
        self.pullingthread = Thread(target=self._pulling_thread_function)
        self.pullingthread.start()

    def stop_pulling_thread(self):
        self.pullingthreadactiveflag = False
        self.pullingthread.join()

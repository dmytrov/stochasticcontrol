import time
import numpy as np

class OptoTrak(object):
    nMarkers = 1
    frameperiod = 1.0
    capturestarttime = 0.0
    captureduration = 10.0
    lastsentframenumber = 0
    getmousecoordsfunc = None
    running = False

    def __init__(self):
        pass

    @staticmethod
    def _GetCurrentFrameNumber():
        dt = time.time() - OptoTrak.capturestarttime
        framenumber = int(dt / OptoTrak.frameperiod)
        return framenumber

    # Transputer

    @staticmethod
    def TransputerDetermineSystemCfg(logfilename):
        return 0

    @staticmethod
    def TransputerInitializeSystem(uFlags):
        return 0

    @staticmethod
    def TransputerLoadSystem(niffilename):
        return 0

    @staticmethod
    def TransputerShutdownSystem():
        return 0

    # Optotrak

    @staticmethod
    def OptotrakActivateMarkers():
        return 0
	
    @staticmethod
    def OptotrakDeActivateMarkers():
        return 0
	
    @staticmethod
    def OptotrakGetCameraParameterStatus():
        return (0, 1, 1, 1, "OK")
    
    @staticmethod
    def OptotrakGetErrorString():
        return (0, "OK")
	
    @staticmethod
    def OptotrakGetStatus():
        return (0, 1, 1, 1, 1, 1.0, 1.0, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1)
	
    @staticmethod
    def OptotrakLoadCameraParameters(filename):
        return 0
	
    @staticmethod
    def OptotrakSaveCollectionToFile(filename):
        return 0
	
    @staticmethod
    def OptotrakSetCameraParameters(nMarkerType, nWaveLength, nModelType):
        return 0
	
    @staticmethod
    def OptotrakSetProcessingFlags(flags):
        return 0
	
    @staticmethod
    def OptotrakSetStroberPortTable(nPort1, nPort2, nPort3, nPort4):
        return 0
	
    @staticmethod
    def OptotrakSetupCollection(nMarkers, fFrameFrequency, 
            fMarkerFrequency, nThreshold, nMinimumGain, nStreamData,
            fDutyCycle, fVoltage, fCollectionTime, fPreTriggerTime, nFlags):
        OptoTrak.nMarkers = nMarkers
        OptoTrak.frameperiod = 1.0 / fFrameFrequency
        OptoTrak.captureduration = fCollectionTime
        OptoTrak.lastsentframenumber = 0
        return 0
	
    @staticmethod
    def OptotrakSetupCollectionFromFile(filename):
        return 0

    # ODAU

    @staticmethod
    def OdauGetStatus(nOdauId):
        return (0, 1, 1, 1, 1, 1.0, 1.0, 1, 1.0, 1.0, 0, 1)
	
    @staticmethod
    def OdauSaveCollectionToFile(filename):
        return 0

    @staticmethod
    def OdauSetupCollection(nOdauId, nChannels, nGain, nDigitalMode,
            fFrameFrequency, fScanFrequency, nStreamData, fCollectionTime,
            fPreTriggerTime, uFlags):
        return 0
	
    @staticmethod
    def OdauSetupCollectionFromFile(filename):
        return 0

    # Real-time data retrieval

    @staticmethod
    def DataGetLatest3D():
        # Blocking
        while not OptoTrak.DataIsReady():
            time.sleep(0.1 * OptoTrak.frameperiod)
        return OptoTrak.DataReceiveLatest3D()
           
    @staticmethod
    def DataGetNext3D():
        return OptoTrak.DataReceiveLatest3D()
	
    @staticmethod
    def DataIsReady():
        return OptoTrak.lastsentframenumber < OptoTrak._GetCurrentFrameNumber()
	
    @staticmethod
    def DataReceiveLatest3D():
        #time.sleep(0.001)
        frameNumber = OptoTrak._GetCurrentFrameNumber()

        bias = np.array([0.0, 0.0, -2500.0])
        factor = 1.0 * np.array([1.0, -1.0, 1.0])
        freq = 0.3
        phases = np.array([freq * np.pi * frameNumber / 60.0, 
                           freq * np.pi * frameNumber / 60.0 + 0.5 * np.pi, 
                           freq * 1.33 * np.pi * frameNumber / 60.0])
        sines = 100.0 * np.vstack([np.sin(phases + np.pi * i/OptoTrak.nMarkers) for i in range(OptoTrak.nMarkers)])
        data = factor * sines + bias 
        
        # Use mouse coordinates if available for marker #1
        if OptoTrak.getmousecoordsfunc is not None:
            mc = OptoTrak.getmousecoordsfunc() 
            data[0, :] = [mc[0], mc[1], 0] 

        OptoTrak.lastsentframenumber = frameNumber
        return (0, frameNumber, 1, 1, data.tolist())
	
    @staticmethod
    def RequestLatest3D():
        return 0
	
    @staticmethod
    def RequestNext3D():
        return 0

    # Buffered data retrieval
    
    @staticmethod
    def DataBufferAbortSpooling():
        return 0
	
    @staticmethod
    def DataBufferInitializeFile(uDataId, filename):
        return 0
	
    @staticmethod
    def	DataBufferSpoolData():
        return (0, 1)
	
    @staticmethod
    def	DataBufferStart():
        OptoTrak.capturestarttime = time.time()
        OptoTrak.lastsentframenumber = 0
        OptoTrak.running = True
        return 0
	
    @staticmethod
    def	DataBufferStop():
        OptoTrak.running = False
        return 0
	
    @staticmethod
    def	DataBufferWriteData():
        realtimedata = OptoTrak.DataIsReady()
        spoolcomplete = (time.time() > OptoTrak.capturestarttime + OptoTrak.captureduration) or not OptoTrak.running
        return (0, realtimedata, spoolcomplete, 1, 100)


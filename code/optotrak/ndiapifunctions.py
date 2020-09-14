import pyndi as ndi


class OptoTrak(object):
    def __init__(self):
        pass

    # Transputer

    @staticmethod
    def TransputerDetermineSystemCfg(logfilename):
        return ndi.TransputerDetermineSystemCfg("{}".format(logfilename))

    @staticmethod
    def TransputerInitializeSystem(uFlags):
        return ndi.TransputerInitializeSystem(uFlags)

    @staticmethod
    def TransputerLoadSystem(niffilename):
        return ndi.TransputerLoadSystem("{}".format(niffilename))

    @staticmethod
    def TransputerShutdownSystem():
        return ndi.TransputerShutdownSystem()

    # Optotrak

    @staticmethod
    def OptotrakActivateMarkers():
        return ndi.OptotrakActivateMarkers()
	
    @staticmethod
    def OptotrakDeActivateMarkers():
        return ndi.OptotrakDeActivateMarkers()
	
    @staticmethod
    def OptotrakGetCameraParameterStatus():
        return ndi.OptotrakGetCameraParameterStatus()
    
    @staticmethod
    def OptotrakGetErrorString():
        return ndi.OptotrakGetErrorString()
	
    @staticmethod
    def OptotrakGetStatus():
        return ndi.OptotrakGetStatus()
	
    @staticmethod
    def OptotrakLoadCameraParameters(filename):
        return ndi.OptotrakLoadCameraParameters("{}".format(filename))
	
    @staticmethod
    def OptotrakSaveCollectionToFile(filename):
        return ndi.OptotrakSaveCollectionToFile("{}".format(filename))
	
    @staticmethod
    def OptotrakSetCameraParameters(nMarkerType, nWaveLength, nModelType):
        return ndi.OptotrakSetCameraParameters(nMarkerType, nWaveLength, nModelType)
	
    @staticmethod
    def OptotrakSetProcessingFlags(flags):
        return ndi.OptotrakSetProcessingFlags(flags)
	
    @staticmethod
    def OptotrakSetStroberPortTable(nPort1, nPort2, nPort3, nPort4):
        return ndi.OptotrakSetStroberPortTable(nPort1, nPort2, nPort3, nPort4)
	
    @staticmethod
    def OptotrakSetupCollection(nMarkers, fFrameFrequency, 
            fMarkerFrequency, nThreshold, nMinimumGain, nStreamData,
            fDutyCycle, fVoltage, fCollectionTime, fPreTriggerTime, nFlags):
        return ndi.OptotrakSetupCollection(nMarkers, fFrameFrequency, 
            fMarkerFrequency, nThreshold, nMinimumGain, nStreamData,
            fDutyCycle, fVoltage, fCollectionTime, fPreTriggerTime, nFlags)
	
    @staticmethod
    def OptotrakSetupCollectionFromFile(filename):
        return ndi.OptotrakSetupCollectionFromFile("{}".format(filename))

    # ODAU

    @staticmethod
    def OdauGetStatus(nOdauId):
        return ndi.OdauGetStatus(nOdauId)
	
    @staticmethod
    def OdauSaveCollectionToFile(filename):
        return ndi.OdauSaveCollectionToFile("{}".format(filename))

    @staticmethod
    def OdauSetupCollection(nOdauId, nChannels, nGain, nDigitalMode,
            fFrameFrequency, fScanFrequency, nStreamData, fCollectionTime,
            fPreTriggerTime, uFlags):
        return ndi.OdauSetupCollection(nOdauId, nChannels, nGain, nDigitalMode,
            fFrameFrequency, fScanFrequency, nStreamData, fCollectionTime,
            fPreTriggerTime, uFlags)
	
    @staticmethod
    def OdauSetupCollectionFromFile(filename):
        return ndi.OdauSetupCollectionFromFile("{}".format(filename))

    # Real-time data retrieval

    @staticmethod
    def DataGetLatest3D():
        return ndi.DataGetLatest3D()
	
    @staticmethod
    def DataGetNext3D():
        return ndi.DataGetNext3D()
	
    @staticmethod
    def DataIsReady():
        return ndi.DataIsReady()
	
    @staticmethod
    def DataReceiveLatest3D():
        return ndi.DataReceiveLatest3D()
	
    @staticmethod
    def RequestLatest3D():
        return ndi.RequestLatest3D()
	
    @staticmethod
    def RequestNext3D():
        return ndi.RequestNext3D()

    # Buffered data retrieval
    
    @staticmethod
    def DataBufferAbortSpooling():
        return ndi.DataBufferAbortSpooling()
	
    @staticmethod
    def DataBufferInitializeFile(uDataId, filename):
        return ndi.DataBufferInitializeFile(uDataId, "{}".format(filename))
	
    @staticmethod
    def	DataBufferSpoolData():
        return ndi.DataBufferSpoolData()
	
    @staticmethod
    def	DataBufferStart():
        return ndi.DataBufferStart()
	
    @staticmethod
    def	DataBufferStop():
        return ndi.DataBufferStop()
	
    @staticmethod
    def	DataBufferWriteData():
        return ndi.DataBufferWriteData()

    @staticmethod
    def	shutdown_on_disconnect():
        OptoTrak.TransputerShutdownSystem()
        return True  # shutdown the applocation


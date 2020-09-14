// LibPyNDI.cpp : Defines the exported functions for the DLL application.
//

#include "ndtypes.h"
#include "ndpack.h"
#include "ndopto.h"

#define BOOST_PYTHON_STATIC_LIB
// The following defines did not help
//#define BOOST_PYTHON_ENABLE_STDCALL
//#define BOOST_PYTHON_ENABLE_CDECL
#include <boost/python.hpp>

#include "LibPyNDI.h"
#include "ImportHelper.h"

Lock LibNDILock;

namespace bp = boost::python;

// Transputer functions
//
WRAP_INT_FUNCTION_STRING(TransputerLoadSystem)
WRAP_INT_FUNCTION_UINT(TransputerInitializeSystem)
WRAP_INT_FUNCTION_STRING(TransputerDetermineSystemCfg)
WRAP_INT_FUNCTION_VOID(TransputerShutdownSystem)


// Optotrak functions
//
WRAP_INT_FUNCTION_STRING(OptotrakLoadCameraParameters)
WRAP_INT_FUNCTION_STRING(OptotrakSetupCollectionFromFile)
int _OptotrakSetupCollection(		   int   nMarkers,
                                       float fFrameFrequency,
                                       float fMarkerFrequency,
                                       int   nThreshold,
                                       int   nMinimumGain,
                                       int   nStreamData,
                                       float fDutyCycle,
                                       float fVoltage,
                                       float fCollectionTime,
                                       float fPreTriggerTime,
                                       int   nFlags )
{
	LIB_NDI_SCOPE_LOCK;
	return OptotrakSetupCollection(	   nMarkers,
                                       fFrameFrequency,
                                       fMarkerFrequency,
                                       nThreshold,
                                       nMinimumGain,
                                       nStreamData,
                                       fDutyCycle,
                                       fVoltage,
                                       fCollectionTime,
                                       fPreTriggerTime,
                                       nFlags);
}
WRAP_INT_FUNCTION_VOID(OptotrakActivateMarkers)
WRAP_INT_FUNCTION_VOID(OptotrakDeActivateMarkers)
bp::tuple _OptotrakGetStatus()
{
	LIB_NDI_SCOPE_LOCK;
	int   res;
	int   nNumSensors = -1;
    int   nNumOdaus = -1;
    int   nNumRigidBodies = -1;
    int   nMarkers = -1;
    float fFrameFrequency = -1.0;
    float fMarkerFrequency = -1.0;
    int   nThreshold = -1;
    int   nMinimumGain = -1;
    int   nStreamData = -1;
    float fDutyCycle = -1.0;
    float fVoltage = -1.0;
    float fCollectionTime = -1.0;
    float fPreTriggerTime = -1.0;
    int   nFlags = -1;
	res = OptotrakGetStatus( 
		&nNumSensors,
        &nNumOdaus,
        &nNumRigidBodies,
        &nMarkers,
        &fFrameFrequency,
        &fMarkerFrequency,
        &nThreshold,
        &nMinimumGain,
        &nStreamData,
        &fDutyCycle,
        &fVoltage,
        &fCollectionTime,
        &fPreTriggerTime,
        &nFlags );
	return boost::python::make_tuple(
		res,
		nNumSensors,
		nNumOdaus,
		nNumRigidBodies,
		nMarkers,
		fFrameFrequency,
		fMarkerFrequency,
		nThreshold,
		nMinimumGain,
		nStreamData,
		fDutyCycle,
		fVoltage,
		fCollectionTime,
		fPreTriggerTime,
		nFlags);
}


bp::tuple _OptotrakGetErrorString()
{
	LIB_NDI_SCOPE_LOCK;
	char errorbuffer[MAX_ERROR_STRING_LENGTH + 1];
	int res = OptotrakGetErrorString(errorbuffer, MAX_ERROR_STRING_LENGTH + 1);
	return bp::make_tuple(res, std::string(errorbuffer));
}

WRAP_INT_FUNCTION_STRING(OptotrakSaveCollectionToFile)

#define STATUS_STRING_LENGTH 10000

bp::tuple _OptotrakGetCameraParameterStatus()
{
	LIB_NDI_SCOPE_LOCK;
	int nCurrentMarkerType = -1;
	int nCurrentWaveLength = -1;
	int nCurrentModelType = -1;
	char szStatus[STATUS_STRING_LENGTH];
	int res = OptotrakGetCameraParameterStatus(
		&nCurrentMarkerType,
		&nCurrentWaveLength,
		&nCurrentModelType,
		szStatus,
		STATUS_STRING_LENGTH);
	return bp::make_tuple(
		res, 
		nCurrentMarkerType,
		nCurrentWaveLength,
		nCurrentModelType,
		std::string(szStatus));
}

int _OptotrakSetCameraParameters(int nMarkerType,
                                 int nWaveLength,
                                 int nModelType)
{
	LIB_NDI_SCOPE_LOCK;
	return OptotrakSetCameraParameters(nMarkerType, nWaveLength, nModelType);
}

WRAP_INT_FUNCTION_UINT(OptotrakSetProcessingFlags)

int _OptotrakSetStroberPortTable(int nPort1,
                                 int nPort2,
                                 int nPort3,
                                 int nPort4)
{
	LIB_NDI_SCOPE_LOCK;
	return OptotrakSetStroberPortTable(nPort1, nPort2, nPort3, nPort4);
}

bp::tuple _OdauGetStatus(int nOdauId)
{
	LIB_NDI_SCOPE_LOCK;
	int      nChannels = -1;
    int      nGain = -1;
    int      nDigitalMode = -1;
    float    fFrameFrequency = -1.0;
    float    fScanFrequency = -1.0;
    int      nStreamData = -1;
    float    fCollectionTime = -1.0;
    float    fPreTriggerTime = -1.0;
    unsigned uCollFlags = 0;
    int      nFlags = -1;
	int res = OdauGetStatus(nOdauId,
							&nChannels,
							&nGain,
							&nDigitalMode,
							&fFrameFrequency,
							&fScanFrequency,
							&nStreamData,
							&fCollectionTime,
							&fPreTriggerTime,
							&uCollFlags,
							&nFlags);
	return bp::make_tuple(
		res,
		nOdauId,
		nChannels,
		nGain,
		nDigitalMode,
		fFrameFrequency,
		fScanFrequency,
		nStreamData,
		fCollectionTime,
		fPreTriggerTime,
		uCollFlags,
		nFlags);
}

WRAP_INT_FUNCTION_STRING(OdauSaveCollectionToFile)
WRAP_INT_FUNCTION_STRING(OdauSetupCollectionFromFile)

int _OdauSetupCollection(int      nOdauId,
                         int      nChannels,
                         int      nGain,
                         int      nDigitalMode,
                         float    fFrameFrequency,
                         float    fScanFrequency,
                         int      nStreamData,
                         float    fCollectionTime,
                         float    fPreTriggerTime,
                         unsigned uFlags )
{
	LIB_NDI_SCOPE_LOCK;
	return OdauSetupCollection(
		nOdauId,
        nChannels,
        nGain,
        nDigitalMode,
        fFrameFrequency,
        fScanFrequency,
        nStreamData,
        fCollectionTime,
        fPreTriggerTime,
        uFlags);
}

#define MAX_NUM_MARKERS 256
static Position3d p3dData[MAX_NUM_MARKERS];
#define MAX_ODAU_CHANNELS 256
static float pODAUData[MAX_ODAU_CHANNELS];

#define MAX_BASIC_DATATYPE_CHANNELS 256
static float pFloatData[MAX_BASIC_DATATYPE_CHANNELS];
static int pIntData[MAX_BASIC_DATATYPE_CHANNELS];
static char pCharData[MAX_BASIC_DATATYPE_CHANNELS];
static double pDoubleData[MAX_BASIC_DATATYPE_CHANNELS];


bp::tuple pPosition3d_to_tuple(Position3d * p3d)
{
	return bp::make_tuple(p3d->x, p3d->y, p3d->z);
}

bp::tuple _DataGetLatest3D()
{
	unsigned int uFrameNumber = 0;
	unsigned int uElements = 0;
	unsigned int uFlags = 0;
	Position3d   *pDataDest = p3dData;
	int res;
	{
		SCOPED_GIL_RELEASE;
		{
			LIB_NDI_SCOPE_LOCK;
			res = DataGetLatest3D(&uFrameNumber, &uElements, &uFlags, (void*)pDataDest);
		}
	}
	bp::list points;
	for (unsigned int i = 0; i < uElements; ++i) 
	{
		points.append(pPosition3d_to_tuple(&pDataDest[i]));
	}
	return bp::make_tuple(res, uFrameNumber, uElements, uFlags, bp::tuple(points));
}

bp::tuple _DataGetNext3D()
{
	unsigned int uFrameNumber = 0;
	unsigned int uElements = 0;
	unsigned int uFlags = 0;
	Position3d   *pDataDest = p3dData;
	int res;
	{
		SCOPED_GIL_RELEASE;
		{
			LIB_NDI_SCOPE_LOCK;
			res = DataGetNext3D(&uFrameNumber, &uElements, &uFlags, (void*)pDataDest);
		}
	}
	bp::list points;
	for (unsigned int i = 0; i < uElements; ++i) 
	{
		points.append(pPosition3d_to_tuple(&pDataDest[i]));
	}
	return bp::make_tuple(res, uFrameNumber, uElements, uFlags, bp::tuple(points));
}

WRAP_INT_FUNCTION_VOID(DataIsReady)

bp::tuple _DataReceiveLatest3D()
{
	unsigned int uFrameNumber = 0;
	unsigned int uElements = 0;
	unsigned int uFlags = 0;
	Position3d   *pDataDest = p3dData;
	int res;
	{
		SCOPED_GIL_RELEASE;
		{	
			LIB_NDI_SCOPE_LOCK;
			res = DataReceiveLatest3D(&uFrameNumber, &uElements, &uFlags, pDataDest);
		}
	}
	bp::list points;
	for (unsigned int i = 0; i < uElements; ++i) 
	{
    	points.append(pPosition3d_to_tuple(&pDataDest[i]));
	}
	return bp::make_tuple(res, uFrameNumber, uElements, uFlags, bp::tuple(points));
}

WRAP_INT_FUNCTION_VOID(RequestLatest3D)
WRAP_INT_FUNCTION_VOID(RequestNext3D)

int _DataBufferInitializeFile(unsigned int  uDataId, std::string szFileName)
{
	LIB_NDI_SCOPE_LOCK;
	return DataBufferInitializeFile(uDataId, (char*)szFileName.c_str());
}

WRAP_INT_FUNCTION_VOID(DataBufferStart)
WRAP_INT_FUNCTION_VOID(DataBufferStop)

bp::tuple _DataBufferSpoolData()
{
	unsigned int  uSpoolStatus;
	int res;
	{
		SCOPED_GIL_RELEASE;
		{
			LIB_NDI_SCOPE_LOCK;
			res = DataBufferSpoolData(&uSpoolStatus);
		}
	}
	return bp::make_tuple(res, uSpoolStatus);
}

bp::tuple _DataBufferWriteData()
{
	unsigned int  uRealtimeData = 0;
	unsigned int  uSpoolComplete = 0;
	unsigned int  uSpoolStatus = 0;
	unsigned long ulFramesBuffered = 0;
	int res;
	{
		SCOPED_GIL_RELEASE;
		{
			LIB_NDI_SCOPE_LOCK;
			res = DataBufferWriteData(&uRealtimeData, &uSpoolComplete, &uSpoolStatus, &ulFramesBuffered);
		}
	}
	return bp::make_tuple(res, uRealtimeData, uSpoolComplete, uSpoolStatus, ulFramesBuffered);
}

WRAP_INT_FUNCTION_VOID(DataBufferAbortSpooling)


int _FileConvert(std::string pszInputFilename,
                 std::string pszOutputFilename,
                 unsigned int  uFileType)
{
	LIB_NDI_SCOPE_LOCK;
	int res = FileConvert((char*)pszInputFilename.c_str(), (char*)pszInputFilename.c_str(), uFileType);
	return res;
}


bp::tuple _FileOpen(std::string pszFilename,
                     unsigned int   uFileId,
                     unsigned int   uFileMode)
{
	LIB_NDI_SCOPE_LOCK;
	int           nItems = -1;
    int           nSubItems = -1;
    long int      lnFrames = -1;
    float         fFrequency = -1.0;
    char          szComments[81];
    void          *pFileHeader;
	int res = FileOpen((char*)pszFilename.c_str(),
                    uFileId,
                    uFileMode,
                    &nItems,
                    &nSubItems,
                    &lnFrames,
                    &fFrequency,
                    szComments,
                    &pFileHeader);
	return bp::make_tuple(res, nItems, nSubItems, lnFrames, fFrequency, std::string(szComments));
}

bp::tuple _FileReadOptotrakFrame(unsigned int  uFileId,
              long int      lnStartFrame,
              unsigned int  nMarkers)
{
	LIB_NDI_SCOPE_LOCK;
	Position3d *pDataDest = p3dData;
	int res = FileRead(uFileId, lnStartFrame, 1, pDataDest);
	bp::list points;
    for (unsigned int i = 0; i < nMarkers; ++i) 
	{
        points.append(pPosition3d_to_tuple(&pDataDest[i]));
    }
	return bp::make_tuple(res, bp::tuple(points));
}

bp::tuple _FileReadODAUFrame(unsigned int  uFileId,
              long int      lnStartFrame,
              unsigned int  nChannels)
{
	LIB_NDI_SCOPE_LOCK;
	float *pDataDest = pODAUData;
	int res = FileRead(uFileId, lnStartFrame, 1, pDataDest);
	bp::list channels;
    for (unsigned int i = 0; i < nChannels; ++i) 
	{
        channels.append(pDataDest[i]);
    }
	return bp::make_tuple(res, bp::tuple(channels));
}


WRAP_INT_FUNCTION_UINT(FileClose)

bp::tuple _FileOpenAll(std::string pszFilename,
                     unsigned int   uFileId,
                     unsigned int   uFileMode)
{
	LIB_NDI_SCOPE_LOCK;
	int           nItems = -1;
    int           nSubItems = -1;
    int           nCharSubItems = -1;
    int           nIntSubItems = -1;
    int           nDoubleSubItems = -1;
    long int      lnFrames = -1;
    float         fFrequency = -1.0;
    char          szComments[81];
    void          *pFileHeader;
	int res = FileOpenAll((char*)pszFilename.c_str(),
                    uFileId,
                    uFileMode,
                    &nItems,
                    &nSubItems,
                    &nCharSubItems,
                    &nIntSubItems,
                    &nDoubleSubItems,
                    &lnFrames,
                    &fFrequency,
                    szComments,
                    &pFileHeader);
	return bp::make_tuple(res, nItems, nSubItems, nCharSubItems, nIntSubItems, nDoubleSubItems, lnFrames, fFrequency, std::string(szComments));
}

bp::tuple _FileReadAllOneFrame(unsigned int  uFileId,
                           long int      lnStartFrame,
                           unsigned int  uNumberOfFloats,
						   unsigned int  uNumberOfChars,
						   unsigned int  uNumberOfInts,
						   unsigned int  uNumberOfDoubles)
{
	LIB_NDI_SCOPE_LOCK;
	float *pDataDestFloat = pFloatData;
	char *pDataDestChar = pCharData;
	int *pDataDestInt = pIntData;
	double *pDataDestDouble = pDoubleData;

	int res = FileReadAll(uFileId, lnStartFrame, 1, pDataDestFloat, pDataDestChar, pDataDestInt, pDataDestDouble);
	bp::list floatChannels;
    for (unsigned int i = 0; i < uNumberOfFloats; ++i) 
	{
        floatChannels.append(pDataDestFloat[i]);
    }
	bp::list charChannels;
    for (unsigned int i = 0; i < uNumberOfChars; ++i) 
	{
        charChannels.append(pDataDestChar[i]);
    }
	bp::list intChannels;
    for (unsigned int i = 0; i < uNumberOfInts; ++i) 
	{
        intChannels.append(pDataDestInt[i]);
    }
	bp::list doubleChannels;
    for (unsigned int i = 0; i < uNumberOfDoubles; ++i) 
	{
        doubleChannels.append(pDataDestDouble[i]);
    }

	return bp::make_tuple(res, bp::tuple(floatChannels), bp::tuple(charChannels), bp::tuple(intChannels), bp::tuple(doubleChannels));
}

WRAP_INT_FUNCTION_UINT(FileCloseAll)



BOOST_PYTHON_MODULE(pyndi)
{
	using namespace boost::python;
	
	// Transputer
	FUNCTION_DEF(TransputerDetermineSystemCfg);
	FUNCTION_DEF(TransputerInitializeSystem);
	FUNCTION_DEF(TransputerLoadSystem);
	FUNCTION_DEF(TransputerShutdownSystem);
    
	// Optotrak
	FUNCTION_DEF(OptotrakActivateMarkers);
	//FUNCTION_DEF(OptotrakChangeCameraFOR);  // no need
	//FUNCTION_DEF(OptotrakConvertRawTo3D);  // no need
	//FUNCTION_DEF(OptotrakConvertTransforms);  // no need
	FUNCTION_DEF(OptotrakDeActivateMarkers);
	FUNCTION_DEF(OptotrakGetCameraParameterStatus);
	FUNCTION_DEF(OptotrakGetErrorString);
	//FUNCTION_DEF(OptotrakGetNodeInfo);  // no need
	FUNCTION_DEF(OptotrakGetStatus);
	FUNCTION_DEF(OptotrakLoadCameraParameters);
	FUNCTION_DEF(OptotrakSaveCollectionToFile);
	FUNCTION_DEF(OptotrakSetCameraParameters);
	FUNCTION_DEF(OptotrakSetProcessingFlags);
	FUNCTION_DEF(OptotrakSetStroberPortTable);
	FUNCTION_DEF(OptotrakSetupCollection);
	FUNCTION_DEF(OptotrakSetupCollectionFromFile);
	
	// ODAU
	FUNCTION_DEF(OdauGetStatus);
	FUNCTION_DEF(OdauSaveCollectionToFile);
	//FUNCTION_DEF(OdauSetAnalogOutputs);  // no need
	//FUNCTION_DEF(OdauSetDigitalOutputs);  // no need
	//FUNCTION_DEF(OdauSetTimer);  // no need
	FUNCTION_DEF(OdauSetupCollection);
	FUNCTION_DEF(OdauSetupCollectionFromFile);

	// Real-time data retrieval
	//FUNCTION_DEF(DataGetLatestCentroid);  // no need
	FUNCTION_DEF(DataGetLatest3D);  // blocking
	//FUNCTION_DEF(DataGetLatestOdauRaw);  // no need
	//FUNCTION_DEF(DataGetLatestRaw);  // no need
	//FUNCTION_DEF(DataGetLatestTransforms);  // no need
	//FUNCTION_DEF(DataGetLatestTransforms2);  // no need
	FUNCTION_DEF(DataGetNext3D);
	//FUNCTION_DEF(DataGetNextCentroid);  // no need
	//FUNCTION_DEF(DataGetNextOdauRaw);  // no need
	//FUNCTION_DEF(DataGetNextRaw);  // no need
	//FUNCTION_DEF(DataGetNextTransforms);  // no need
	//FUNCTION_DEF(DataGetNextTransforms2);  // no need
	FUNCTION_DEF(DataIsReady);  // non-blocking
	FUNCTION_DEF(DataReceiveLatest3D);  // non-blocking
	//FUNCTION_DEF(DataReceiveLatestCentroid);  // no need
	//FUNCTION_DEF(DataReceiveLatestOdauRaw);  // no need
	//FUNCTION_DEF(DataReceiveLatestRaw);  // no need
	//FUNCTION_DEF(DataReceiveLatestTransforms);  // no need
	//FUNCTION_DEF(DataReceiveLatestTransforms2);  // no need
	//FUNCTION_DEF(ReceiveLatestData);  // no need, deprecated
	FUNCTION_DEF(RequestLatest3D);  // non-blocking
	//FUNCTION_DEF(RequestLatestCentroid);  // no need
	//FUNCTION_DEF(RequestLatestOdauRaw);  // no need
	//FUNCTION_DEF(RequestLatestRaw);  // no need
	//FUNCTION_DEF(RequestLatestTransforms);  // no need
	FUNCTION_DEF(RequestNext3D);
	//FUNCTION_DEF(RequestNextCentroid);  // no need
	//FUNCTION_DEF(RequestNextOdauRaw);  // no need
	//FUNCTION_DEF(RequestNextRaw);  // no need
	//FUNCTION_DEF(RequestNextTransforms);  // no need
	
	// Buffered data retrieval
	FUNCTION_DEF(DataBufferAbortSpooling);
	FUNCTION_DEF(DataBufferInitializeFile);
	//FUNCTION_DEF(DataBufferInitializeMem);   // no need
	FUNCTION_DEF(DataBufferSpoolData);  // non-blocking
	FUNCTION_DEF(DataBufferStart);  // non-blocking
	FUNCTION_DEF(DataBufferStop);  // non-blocking
	FUNCTION_DEF(DataBufferWriteData);  // non-blocking

	// File IO
	FUNCTION_DEF(FileOpen);
	FUNCTION_DEF(FileReadOptotrakFrame);
	FUNCTION_DEF(FileReadODAUFrame);
	FUNCTION_DEF(FileClose);
	FUNCTION_DEF(FileOpenAll);
	FUNCTION_DEF(FileReadAllOneFrame);
	FUNCTION_DEF(FileCloseAll);

}


import sys
import time
import pyndi as ndi

try:
    res = ndi.TransputerLoadSystem("system")
    if res != 0: raise Exception(res)
    time.sleep(1)

    OPTO_LOG_ERRORS_FLAG = 0x0001
    res = ndi.TransputerInitializeSystem( OPTO_LOG_ERRORS_FLAG )
    if res != 0: raise Exception(res)

    res = ndi.OptotrakLoadCameraParameters( "standard" )
    if res != 0: raise Exception(res)

    #res = ndi.OptotrakSetStroberPortTable(2, 0, 0, 0)
    #if res != 0: raise Exception(res)

    NUM_MARKERS = 8
    OPTOTRAK_BUFFER_RAW_FLAG = 0x0020
    OPTOTRAK_GET_NEXT_FRAME_FLAG = 0x2000

    res = ndi.OptotrakSetupCollection(
            NUM_MARKERS,    #/* Number of markers in the collection. */
            100.0,    #/* Frequency to collect data frames at. */
            2500.0,  #/* Marker frequency for marker maximum on-time. */
            30,             #/* Dynamic or Static Threshold value to use. */
            160,            #/* Minimum gain code amplification to use. */
            1,              #/* Stream mode for the data buffers. */
            0.35,     #/* Marker Duty Cycle to use. */
            7.0,     #/* Voltage to use when turning on markers. */
            10.0,     #/* Number of seconds of data to collect. */
            0.0,     #/* Number of seconds to pre-trigger data by. */
            OPTOTRAK_BUFFER_RAW_FLAG | OPTOTRAK_GET_NEXT_FRAME_FLAG ) 
    if res != 0: raise Exception(res)
    time.sleep(1)

    res = ndi.OptotrakActivateMarkers()
    if res != 0: raise Exception(res)

    OPTOTRAK = 0
    res = ndi.DataBufferInitializeFile( OPTOTRAK, "R#001.S05" )
    if res != 0: raise Exception(res)

    res = ndi.DataBufferStart()
    if res != 0: raise Exception(res)
    res = ndi.RequestLatest3D()
    if res != 0: raise Exception(res)

    uSpoolComplete = False
    while not uSpoolComplete:
        (res, uRealtimeDataReady, uSpoolComplete,
            uSpoolStatus, nFrames) = ndi.DataBufferWriteData()
        if res != 0: raise Exception(res)   
        if uRealtimeDataReady:
            (res, uFrameNumber, uElements, uFlags,
                p3dData) = ndi.DataReceiveLatest3D()
            if res != 0: raise Exception(res)
            print("Frame Number: ", uFrameNumber)
            print("Elements    : ", uElements)
            print("Flags       : ", uFlags)
            print p3dData
            res = ndi.RequestLatest3D()
            if res != 0: raise Exception(res)
            uRealtimeDataReady = 0

finally:
    print ndi.OptotrakGetErrorString()
    print "Shutting down..."
    ndi.TransputerShutdownSystem()



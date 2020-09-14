import os
import numpy as np
import optotrak.pyndi as ndi
from optotrak.ndiapiconstants import NDI as ndiconst
import matplotlib.pyplot as plt



def read_OPTO_file(filenameOPTO):
    infile = 1
    res = ndi.FileOpenAll(filenameOPTO, infile, ndiconst.OPEN_READ)
    retcode, nitems, nfloats, nchars, nints, ndoubles, nframes, frequency, comment = res
    if retcode != 0:
        retcode, errorstring = ndi.OptotrakGetErrorString()
        raise IOError(errorstring)
    markerdata = []
    for i in range(nframes):
        res = ndi.FileReadAllOneFrame(infile, i, nitems * nfloats, nchars, nints, ndoubles)
        retcode, floatdataframe, chardataframe, intdataframe, doubledataframe = res
        markerdata.append(floatdataframe)
    ndi.FileCloseAll(infile)
    return np.array(markerdata, dtype=np.dtype('f4'))



def read_ODAU_file(filenameODAU):
    infile = 1
    res = ndi.FileOpenAll(filenameODAU, infile, ndiconst.OPEN_READ)
    retcode, nitems, nfloats, nchars, nints, ndoubles, nframes, frequency, comment = res
    if retcode != 0:
        retcode, errorstring = ndi.OptotrakGetErrorString()
        raise IOError(errorstring)
    analogdata = []
    for i in range(nframes):
        res = ndi.FileReadAllOneFrame(infile, i, nfloats, nchars, nints, ndoubles)
        retcode, floatdataframe, chardataframe, intdataframe, doubledataframe = res
        analogdata.append(intdataframe)
    ndi.FileCloseAll(infile)
    return np.array(analogdata, dtype=np.int16)


def convert_OPTO_to_numpy(filename):
    data = read_OPTO_file(filename)
    np.save(filename + ".npy", data)

def convert_ODAU_to_numpy(filename):
    data = read_ODAU_file(filename)
    np.save(filename + ".npy", data)

def convert_all_files_in_filesystem_subtree(path):
    diritems = os.listdir(path)
    for diritem in diritems:
        diritempath = path + "/" + diritem
        if os.path.isdir(diritempath):
            convert_all_files_in_filesystem_subtree(diritempath)
        else:
            if diritempath[-5:] == ".OPTO":
                print("Converting OPTO file '{}'...".format(diritempath))
                convert_OPTO_to_numpy(diritempath)
                print(diritem + ".npy file is saved.")
            elif diritempath[-5:] == ".ODAU":
                print("Converting ODAU file '{}'...".format(diritempath))
                convert_ODAU_to_numpy(diritempath)
                print(diritem + ".npy file is saved.")



if __name__ == "__main__":
    convert_all_files_in_filesystem_subtree("../../recordings")


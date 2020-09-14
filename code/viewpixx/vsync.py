"""
Strobe vertical sync pulse at digital output pin 1
"""
import time
from pypixxlib.viewpixx import VIEWPixx3D

vpdevice = VIEWPixx3D()
print(vpdevice.name)
print(vpdevice.subsystems)
print("Firmware version: {}, required >= 31".format(vpdevice.firmware_revision))

dout = vpdevice.dout
vpdevice.updateRegisterCache()
print("dout.getScheduleRunningState: {}".format(dout.getScheduleRunningState()))
print("Stoppting all schedules...")
dout.stopSchedule()
vpdevice.updateRegisterCache()
print("dout.getScheduleRunningState: {}".format(dout.getScheduleRunningState()))

dout.disablePixelMode()
vpdevice.writeRegisterCache()
print("dout.isPixelModeEnabled: {}".format(dout.isPixelModeEnabled()))

print("dout.getNbrOfBit: {}".format(dout.getNbrOfBit()))

buffaddr = 0
data = [1, 0, 0, 0]
datasize = len(data)

dout.setScheduleCountDown(False)  # disable countdown
dout.setScheduleRate(datasize, "video")  # number of samples sent to dout from the RAM buffer per video frame
dout.setScheduleOnset(0)  # ns delay from the onset trigger
vpdevice.writeRegisterCache()

print("Writting data to ViewPixx RAM...")
vpdevice.writeRam(buffaddr, data)
ramchunk = vpdevice.readRam(buffaddr, datasize)
print("vpdevice.readRam: {}".format(ramchunk))

dout.setBaseAddress(buffaddr)  # RAM data buffer start
dout.setBufferSize(2 * datasize)  # RAM data buffer size. Must be 2 * data size
dout.setReadAddress(buffaddr)  # set the internal RAM read pointer, it may be initialized by garbage
vpdevice.writeRegisterCache()

print("dout.getReadAddress: {}".format(dout.getReadAddress()))
print("dout.getBaseAddress: {}".format(dout.getBaseAddress()))
print("dout.getBufferSize: {}".format(dout.getBufferSize()))
print("dout.getScheduleUnit: {}".format(dout.getScheduleUnit()))
print("dout.getScheduleRate: {}".format(dout.getScheduleRate()))

print("Starting the schedule...")
vpdevice.writeRegisterCache()
dout.startSchedule()
vpdevice.updateRegisterCache()
print("dout.getScheduleRunningState: {}".format(dout.getScheduleRunningState()))

vpdevice.close()


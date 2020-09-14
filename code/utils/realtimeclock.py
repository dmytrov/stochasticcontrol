"""
Provides high-precision relative timastamp clock/timer
"""
import time
import platform

os_system = platform.system()
if os_system == "Linux":
    highrestimer = time.clock
elif os_system == "Windows":
    highrestimer = time.clock

rtc = time.time

def timer_resolution(timer_function):
    t0 = timer_function()
    t1 = timer_function()
    while t0 == t1:
        t1 = timer_function()
    return t1-t0
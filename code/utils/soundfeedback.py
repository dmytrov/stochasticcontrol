import os
import time
import psychopy.sound as pps
import numpy as np

module_path = os.path.dirname(os.path.abspath(__file__))
sounds_path = os.path.join(module_path, "../resources/sounds/")

dorbell = pps.Sound(value=os.path.join(sounds_path, "dorbell.wav"))
air_horn = pps.Sound(value=os.path.join(sounds_path, "air_horn.wav"))
buzzer = pps.Sound(value=os.path.join(sounds_path, "buzzer.wav"))
oink = pps.Sound(value=os.path.join(sounds_path, "oink.wav"))
pain = pps.Sound(value=os.path.join(sounds_path, "pain.wav"))
screaming = pps.Sound(value=os.path.join(sounds_path, "screaming.wav"))
sms_alert = pps.Sound(value=os.path.join(sounds_path, "sms_alert.wav"))

negatives = [dorbell, air_horn, buzzer, oink, pain]

def negative():
    i = np.random.randint(low=0, high=len(negatives), size=1)[0]
    negatives[i].play()

def positive():
    sms_alert.play()


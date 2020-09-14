# Delayed Feedback Experiment
Author: Dmytro Velychko <velychko@staff.uni-marburg.de>

University of Marburg, Department of Psychology

AE Endres, Theoretical Neuroscience Lab.

# Prerequisites. Installing python serial port library in Windows

Python comes with a handy package manager called `pip`. It is a command line utility. Command line in Windows OS is accessible through `cmd.exe`.
Check out the [Wikipedia article on pip](https://en.wikipedia.org/wiki/Pip_(package_manager)). 

To install or uninstall a package, simply run
```
pip install some-package-name
pip uninstall some-package-name
```

Python serial port library is called `pyserial`. To install it, run
```
pip install pyserial
```
Wait till pip downloads and installs the library.

If the computer doesn't have an Internet connection, then download `pyserial‑3.3‑py2.py3‑none‑any.whl` library from [http://www.lfd.uci.edu/~gohlke/pythonlibs/](http://www.lfd.uci.edu/~gohlke/pythonlibs/)
and install it with pip:

```
pip install pyserial‑3.3‑py2.py3‑none‑any.whl
```

# Running the delayed feedback experiment

## Optotrak 
To use the Optotrak Certus, turn on all three its components: the main control unit, ODAU (digital and analog input unit), and the camera unit. You should hear beeping sounds it makes at the startup.

## Instructing the participant

```
Delayed Feedback Experiment
Instruction for the Participant

We are interested in how humans control their movements. Particularly, how the movement changes when the feedback from the environment changes.

Yous task is to perform reaching movements. You will be controlling a small white circle at the screen. The reaching has to be performed from the starting position to the targets. The starting position is marked with a small Velcro pad on the table, keep your left index finger there to locate it quickly.

Every reaching trial has the following phases:
- A large gray circle appears at your starting position. During this phase, the controlled circle is not visible. You will have to move your right index finger to the Velcro pad. Once your finger is there, the gray circle disappears.
- Shortly after a green dot appears. This is the target you have to reach. Move your right arm to place the white circle over the target. When half of the allowed time has elapsed, the controlled circle turns red.
- Depending whether you have reached or missed the target, you get a positive of a negative auditory feedback. 
You really want to avoid the negative feedback, it is very unpleasant!
```



## EMG
Ask Prof. Dr. Gunner Blohm for a quick tutorial.

Locate the active muscles and connect the EMG ground and pickup electrodes to the participant. 
Clean the pickup electrodes and the skin under the electrodes with an alcohol pad.

Ground electrode can be connected to the opposite arm elbow.
It can be reused for the same participant many times, just write the participant's name on it. The adhesive interface tape on the pickup electrodes is usually not reusable.

While connecting, the EMG speaker (beeper) can be turned off. It should be on while running the experiment, so you could notice if an electrode contact gets worse.

Make sure the wires don't constrain the arm motion.

## Optotrak marker
Place one Optotrak marker, connected to the port 1 of the strober, at the end of the right index finger. Ensure it faces the camera when the participant performs reaching motions. Secure it with a sticky tape. Secure the marker wires.

## Placing the participant
The participant should sit in the chair. Adjust the height such that reaching arm motion is easy to perform and the whole screen is visible. The left arm index finger should be placed at the Velcro pad, it makes it easy to reach the staring position, as there is no visual feedback during that phase.

## Auditory feedback
Subjects are presented with positive and negative auditory feedback signals after every reaching trial.

## Running the experiment script
Login into the Optotrak computer under the `velychko` account, use the usual lab password.

The code for the experiment is in `code` directory at the windows desktop. The recorded data is stored in a timestamped folder in the `recordings` directory, also at the desktop.

The main python script for the experiment is `code/experiments/delayedfeedback.py`.
It has a number of parameters and provides a minimal help if run as
`python -O delayedfeedback.py --help`:

```
usage: delayedfeedback.py [-h] [--v] [--b B] [--n N] [--t T] [--d D]

Delayed feedback experiment

optional arguments:
-h, --help show this help message and exit
--v show program's version number and exit
--b B Run only B-th block, 0 for all blocks. Default is 0
--n N Full number of blocks. Default is 3
--t T Feedback time shift, range [-1.0, 1.0] sec. Default is 00.
--d D Digital IO device number. Default is 0 (first)

Unauthorized copying and/or use of this software is strictly prohibited.
This software comes with no warranty.
Written by Dmytro Velychko <velychko@staff.uni-marburg.de>
University of Marburg, Department of Psychology
AE Endres, Theoretical Neuroscience Lab.

Available serial ports: ['/dev/ttyUSB0']
```

To check the experiment script version, run `python delayedfeedback.py --v`. The output should look like
```
delayedfeedback.py 0.1
```
So far the latest version is 0.3, but it may be updated later. This is to check whether you actually have the latest version at the lab computer.

It also enumerates available serial ports in the system. Serial ports are assigned integer indexes starting from 0 in the order they appear in the list. In this case, only one such port is present in the list, named `'/dev/ttyUSB0'`, with index `0`. To use a specific serial port, provide its index after the `--d` parameter name, like `--d 0` or `--d 1`.


### Blocks

The experiment can be run in one or multiple blocks. We have decided to run it in 3 blocks for every of three feedback delays, 9 blocks in total. Each block takes 10-15 minutes.

Here are the commands to run the full set of blocks:
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2
```

Each participant has to have its own order of blocks. 

#### Order of the blocks

1. Participant 1, 7, 13, 19, 25
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2
```

2. Participant 2, 8, 14, 20, 26
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1
```

3. Participant 3, 9, 15, 21, 27
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2
```

4. Participant 4, 10, 16, 22, 28
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0
```

5. Participant 5, 11, 17, 23, 29
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1
```

6. Participant 6, 12, 18, 24, 30
```
python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.2
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.2

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.1
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.1

python -O delayedfeedback.py --d 0 --n 3 --b 1 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 2 --t -0.0
python -O delayedfeedback.py --d 0 --n 3 --b 3 --t -0.0
```


###  Prepare checklist
- turn the Optotrak on, all three units
- make a test run of the script to check the Optotrak
- check audio feedback works 
- clean the electrodes and skin areas with alcohol
- turn on the EMG amplifier
- check the amplification is 10000
- turn off the EMG beeper
- connect the ground electrode
- connect the EMG pick-up electrodes
- secure the EMG cables with sticky tape
- check EMG the beeper is on
- position the participant, adjust the height
- check the LED marker is visible from the camera
- assign an incrementing ID to the subject
- make a note in the info.txt for the start time of the recording


### Run checklist
- turn the EMG beeper on
- copy-paste a command line for the script, run
- check the optotrak initialized correctly and the experiment runs
- wait for the block completion
- repear for the next block

### Aborting the experiment
If you have to abort the experiment before the trial completes, click on graphical window of the experiment (usually it is on the right screen, which is the PixxView monitor, you can't see it, so just move the mouse pointer there and click), and press `q` to quit.

# Potential problems

### Failed to start
Sometimes Optotrak refuses connection or return an error during the initialization phase. Just re-run the script.

### Frame drops
Occasionally Optotrak can't provide real-time data quickly enough, so you may see warning messages at the console (`Frame X: SKIPPED FRAMES: Y`). most of the time it is 1-2 frames. If it doesn't exceed ~20, then just ignore it. If it drops many frames very often, then maybe reboot (off/on) the Optotrak, re-run the block, make a note. I can't help here much. Also so far it hasn't happened to me yet.

### Calibration
The Optotrak-to-screen position can be re-calibrated if necessary. Delete the `code/experiments/projection_calibration.pkl` file and run `python -O delayedfeedback.py`. You will go through the calibration sequence. Move the marker to the red dots as they appear on the screen. Once finished, the new calibration is stored.


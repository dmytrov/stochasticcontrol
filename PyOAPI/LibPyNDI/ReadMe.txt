========================================================================
    DYNAMIC LINK LIBRARY : LibPyNDI Project Overview
========================================================================
NDI OptoTRACK python wrapper module for Windows.


Created by: Dmytro Velychko
	dmytro.velychko@gmail.com		

Requires:
	Windows 7+
	Visual Studio Express for Desktop 2012
	python 2.7 32-bit
	boost 1.63.0 32-bit. Manually compiled. Run "b2 address-model=32"

Environment variables:
	LIBBOOSTDIR = C:\boost_1_63_0
	PYTHONDIR = C:\Python27

After compilation run PyOAPI\Release\make_pyd.bat to get the pyndi.pyd module.


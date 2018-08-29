#!/bin/bash
SRC="SPI.cpp Palettes.cpp ocv-gui.cpp ocv-lepton-threads.cpp ocv-main.cpp"

LIB_OCV="-I /usr/local/include/opencv -L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio"
LIB_FLIR="-LleptonSDKEmb32PUB/Debug  -lLEPTON_SDK" 
LIB="-I . -L/usr/lib/arm-linux-gnueabihf  -pthread" 

_OPTIMIZATIONS="-mtune=arm6 -fforward-propagate -finline-functions"

EXE="ocv-raspberry-video" 

g++ -D __DEBUG  $SRC -o $EXE $LIB $LIB_OCV $LIB_FLIR $OPTIMIZATIONS


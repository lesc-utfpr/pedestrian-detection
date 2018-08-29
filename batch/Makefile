CC = g++
CFLAGS = -g -Wall
SRCS = Saliency_CNN.cpp
PROG = HelloWorld

INC_CAFFE = -I/home/diulhio/caffe/include
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) $(INC_CAFFE) -o $(PROG) $(SRCS) $(LIBS)

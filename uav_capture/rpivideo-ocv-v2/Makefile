# Source, executable, includes, libraries for the project
PRJ_NAME = ocv-raspberry-video
INCL = 
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
D = $(SRC:.cpp=.d)

INC_OCV=-I /usr/local/include/opencv -I/usr/local/include/raspicam
LIB_OCV=-L /usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lraspicam -lraspicam_cv
LIB_FLIR=-LleptonSDKEmb32PUB/Debug  -lLEPTON_SDK 
LIB=-L/usr/lib/arm-linux-gnueabihf  -pthread

EXE = $(PRJ_NAME)
LSS = $(PRJ_NAME)
SIZEDUMMY = 


# Tools
C = gcc
CC = g++
AR = ar
OBJCOPY = objcopy
OBJDUMP = objdump
RM = /bin/rm -f

# Compiler and Linker Defines
CPP_FLAGS = -c -w -D__DEBUG
C_FLAGS   = 
INC_PATH = -I. $(INC_OCV)
LIBPATH = -L. $(LIB) $(LIB_OCV) $(LIB_FLIR)
LD_FLAGS = $(LIBPATH)
CFDEBUG = 



all: $(D) $(EXE)

# Building Arduino core library

# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ)
	@echo ' '
	@echo 'Linking object files to create: '$@ 
	@$(CC) -w -o $@ $^ $(LD_FLAGS)

# Compile and Assemble C Source Files into Object Files
%.o: %.cpp
	@echo 'Compiling project file: ' $<
	@$(CC) $(CPP_FLAGS) $(INC_PATH)  $< -o $@ $(LD_FLAGS)

-include $(SRC:.cpp=.d)
%.d: %.cpp
	@echo 'Generating dependencies for: ' $<
	@$(CC) $(CPP_FLAGS) $(INC_PATH) $< -MM -MT '$*.o $*.d ' -MD



# Clean Up Objects, Executables, dumps out of source directory
clean:
	@echo ' '
	@echo ' '
	$(RM) $(OBJ) $(EXE) $(D) $(EXE)

	
	
mawdebug:
	@echo 'Arduino core contains:'
	@echo $(ARD_CORE_SRC_CPP)
	@echo ' '
	@echo ' '
	@echo $(ARD_CORE_OBJS_CPP)
	@echo ' '
	@echo ' '
	@echo $(ARD_CORE_SRC_C)
	@echo ' '
	@echo ' '
	@echo $(ARD_CORE_OBJS_C)
	@echo ' '
	@echo ' '
	@echo $(SRC)

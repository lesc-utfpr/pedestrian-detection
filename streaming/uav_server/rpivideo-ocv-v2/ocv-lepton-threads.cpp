#include "ocv-lepton-threads.h"
#include "ocv-utils.h"


using namespace cv;
using namespace std;

#define FLIR

// global variables
uint8_t flirData[PACKET_SIZE*PACKETS_PER_FRAME];
pthread_mutex_t mutex_flirData; // controls the concurrent access to FLIR data
uint16_t *flirFrameBuffer = (uint16_t *)flirData;
uint16_t minValue = FLIR_MAX_VALUE;
uint16_t maxValue = FLIR_MIN_VALUE;

long int frameControl = 0, frameControl2 = 0;


CvVideoWriter *videoWriter;

/*****************************
 *****************************/
void *readDataFromFLIR(void *ptr) {


	while (!app_exit) {
#ifdef DEBUG
		printf("FLIR sensor is being read...\n");
#endif
#ifdef DEBUG
		printf("Reading SPI...");
#endif
		//if (app_exit) continue;
		
		// entering critical region
		pthread_mutex_lock(&mutex_flirData); 
		
		//read data packets from lepton over SPI
		int resets = 0;
		for(int j=0;j<PACKETS_PER_FRAME;j++) {
			//if it's a drop packet, reset j to 0, set to -1 so it'll be at 0 again loop
#ifdef DEBUG
			printf("\nReading packet # %d", j);
#endif
			// reading the next packet
			read(spi_cs0_fd, flirData+sizeof(uint8_t)*PACKET_SIZE*j, sizeof(uint8_t)*PACKET_SIZE);
			int packetNumber = flirData[j*PACKET_SIZE+1];
			// checking if it is the correct packet from the current frame
			if(packetNumber != j) {
				j = -1;
				resets += 1;
				usleep(1000);
				//Note: we've selected 750 resets as an arbitrary limit, since there should never be 750 "null" packets between two valid transmissions at the current poll rate
				//By polling faster, developers may easily exceed this count, and the down period between frames may then be flagged as a loss of sync
				if(resets == 750) {
					SpiClosePort(0);
					usleep(750000);
					SpiOpenPort(0);
				}
			}
		}
		// leaving critical region
		//pthread_mutex_unlock(&mutex_flirData); 
#ifdef DEBUG
		printf("done.\n");
#endif
		if(resets >= 30) {
			printf("Number of resets: %d\n", resets);
		}
		
		//if (app_exit) continue;

		uint16_t value;
		for(int i=0;i<FRAME_SIZE_UINT16;i++) {
			//skip the first 2 uint16_t's of every packet, they're 4 header bytes
			if(i % PACKET_SIZE_UINT16 < 2) {
				continue;
			}
			
			// entering critical region
			//pthread_mutex_lock(&mutex_flirData); 
			//flip the MSB and LSB at the last second
			int temp = flirData[i*2];
			flirData[i*2] = flirData[i*2+1];
			flirData[i*2+1] = temp;
			
			value = flirFrameBuffer[i];
			if(value > maxValue) {
#ifdef SINGLE_MAX_MIN_VALUE
				// this implementation has a fixed range of value for all images built from
				// the sensor data, in order to have a unique color scale for the whole video
				// the range of value increases smoothly and remains stable during the whole video
				
				if (maxValue == FLIR_MIN_VALUE) {
					// plus 5% 5% to decrease the chance of abrupt variance
					maxValue = value + (value/20);
					printf("MIN = %d\t\t*MAX* = %d\n", minValue, maxValue);
				}
				else {
					//maxValue = (maxValue + value + (value/10))/2; 
					int actualRange = maxValue - minValue;
					int newRange = value - minValue;
					int diff = newRange - actualRange;
					printf("V=%d\tA=%d\tN=%d\tDIFF = %d\n", value, actualRange, newRange, diff);
					if (diff < (actualRange/20)) {
						maxValue = value; //+ (value/10);
						printf("MIN = %d\t\t*MAX* = %d\n", minValue, maxValue);
					}
				}
#else
				// this implementation variable range value for all images built from
				// the sensor data. In other words, the range is stablished for each frame,
				// and hence, the color scale varies during the video
				// the range of values may change at each frame
				maxValue = value;
				printf("MIN = %d\t\t*MAX* = %d\n", minValue, maxValue);
#endif
			}
			if(value < minValue) {
#ifdef SINGLE_MAX_MIN_VALUE
				// this implementation has a fixed range of value for all images built from
				// the sensor data, in order to have a unique color scale for the whole video
				// the range of value increases smoothly and remains stable during the whole video
								
				if (minValue == FLIR_MAX_VALUE) {
					// minus 5% to decrease the chance of abrupt variance
					minValue = value - (value/20);
					printf("*MIN* = %d\t\tMAX = %d\n", minValue, maxValue);
				}
				else {
					//minValue = (minValue - value - (value/10))/2;
					int actualRange = maxValue - minValue;
					int newRange = maxValue - value;
					int diff = newRange - actualRange;
					if (diff < (actualRange/20)) {
						minValue = value; //- (value/10);
						printf("*MIN* = %d\t\tMAX = %d\n", minValue, maxValue);
					}
				}
#else
				// this implementation variable range value for all images built from
				// the sensor data. In other words, the range is stablished for each frame,
				// and hence, the color scale varies during the video
				// the range of values may change at each frame
				
				minValue = value;
				printf("MIN = %d\t\t*MAX* = %d\n", minValue, maxValue);
#endif
			}
			// leaving critical region
			//pthread_mutex_unlock(&mutex_flirData); 
		}
		// leaving critical region
		pthread_mutex_unlock(&mutex_flirData); 
		// wait 1ms to alleviate the CPU processing
		SLEEP_1_MS;
#ifdef DEBUG
		printf("FLIR data has been completely read.\n");
#endif
	}
	// assuring the mutex will be released before exiting
	pthread_mutex_unlock(&mutex_flirData); 
	
	pthread_exit(NULL);
}

/***********************************
 ***********************************/
void *displayFLIRDataAsImage(void *ptr) {
	


	while (!app_exit) {
#ifdef DEBUG
		printf("Building image...");
#endif
		// entering critical region
#ifdef DISPLAY_MUTEX_FULL_LOCK
		pthread_mutex_lock(&mutex_flirData); 
		pthread_mutex_lock(&mutex_FLIRImage); 
#endif
		for(int i=0;i<FRAME_SIZE_UINT16;i++) {
			if(i % PACKET_SIZE_UINT16 < 2) {
				continue;
			}

#ifndef DISPLAY_MUTEX_FULL_LOCK
			pthread_mutex_lock(&mutex_flirData); 
#endif
			float diff = maxValue - minValue;
			float scale = 255/diff;
			uint16_t value;
			value = (flirFrameBuffer[i] - minValue) * scale;
#ifndef DISPLAY_MUTEX_FULL_LOCK
			pthread_mutex_unlock(&mutex_flirData); 
#endif
			
			//const int *colormap = colormap_rainbow;
			const int *colormap = colormap_grayscale;
			//const int *colormap = colormap_ironblack;
			
			int row, column;
			column = (i % PACKET_SIZE_UINT16 ) - 2;
			row = i / PACKET_SIZE_UINT16;

#ifndef DISPLAY_MUTEX_FULL_LOCK
			pthread_mutex_lock(&mutex_FLIRImage); 
#endif
			RgbImage  imgA(mwFLIRImage);
			imgA[row][column].b = colormap[3*value+2];
			imgA[row][column].g = colormap[3*value+1];
			imgA[row][column].r = colormap[3*value];
			
#ifndef DISPLAY_MUTEX_FULL_LOCK
			pthread_mutex_unlock(&mutex_FLIRImage); 
#endif
		}
		// leaving critical region
#ifdef DISPLAY_MUTEX_FULL_LOCK
		pthread_mutex_unlock(&mutex_FLIRImage); 
		pthread_mutex_unlock(&mutex_flirData); 
		// wait 1ms to alleviate the CPU processing
		SLEEP_1_MS;
#endif
#ifdef DEBUG
		printf("done.\n");
#endif
	}
	// assuring that all mutexes are released before exiting
	pthread_mutex_unlock(&mutex_FLIRImage); 
	pthread_mutex_unlock(&mutex_flirData); 
	
	pthread_exit(NULL);
}


/***********************************
 ***********************************/
void *writeFLIRVideo(void *ptr) {
	printf("\nThread FLIR started...\n");

	/** SOCKET **/
	int localSocket,
        remoteSocket,
        port = 4098;                               

    struct  sockaddr_in localAddr,
                        remoteAddr;
    pthread_t thread_id;
    
           
    int addrLen = sizeof(struct sockaddr_in);
    
    localSocket = socket(AF_INET , SOCK_STREAM , 0);
    if (localSocket == -1){
         perror("socket FLIR failed!!");
    } 
    
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = INADDR_ANY;
    localAddr.sin_port = htons( port );
    
    if( bind(localSocket,(struct sockaddr *)&localAddr , sizeof(localAddr)) < 0) {
         perror("bind FLIR socket");
         exit(1);
    }
    
    listen(localSocket , 3);
    
    /************/
    
    Mat img;
    img = Mat::zeros(60 , 80, CV_8UC3);  
    if (!img.isContinuous()) {
        img = img.clone();
    }
    int imgSize = img.total() * img.elemSize();
    int bytes = 0;
    

	long int lastFrame = 0;
	
	while (!app_exit) {		
		
		std::cout << "FLIR waiting connection..." << std::endl;
		remoteSocket = accept(localSocket, (struct sockaddr *)&remoteAddr, (socklen_t*)&addrLen);
		
		if (remoteSocket < 0) {
			perror("accept failed!");
			exit(1);
		} 
		else
		{
			while(1)
			{					
				if(frameControl > lastFrame){
					
#ifndef DISPLAY_MUTEX_FULL_LOCK
					pthread_mutex_lock(&mutex_FLIRImage); 
#endif
					img = cvarrToMat(mwFLIRImage);
					if ((bytes = send(socket, img.data, imgSize, 0)) < 0){
						 std::cerr << "bytes = " << bytes << std::endl;
						 break; // Não enviou mais, então sai e espera outra conexão
					}                
					// Escrever mwFLIRImage
					
#ifndef DISPLAY_MUTEX_FULL_LOCK
					pthread_mutex_unlock(&mutex_FLIRImage); 
#endif
					gettimeofday(&last_frame, NULL);
					printf("FLIR control: %ld\n",frameControl);
					lastFrame = frameControl;
					frameControl2 ++;
				}
				
			} // fim while -> sai no break caso nao envie...
		}
	}
	pthread_exit(NULL);
}













/***********************************
 ***********************************/
void *writeRGBVideo(void *ptr) {
	frameControl = 0;
	frameControl2 = 0;
	
	printf("\nThread RGB started...\n");
	
	long int lastFrame = 0;
	
	
	raspicam::RaspiCam_Cv Camera;
    
    cv::Mat image;
    //Camera.set( CV_CAP_PROP_FORMAT, CV_8UC1 );
    Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 720  );
    Camera.set( CV_CAP_PROP_FRAME_WIDTH, 1280  );
    Camera.set( CV_CAP_PROP_FPS , 10);

	printf("Opening Camera...\n");
    if (!Camera.open()) {printf("Error opening the camera\n");}
    
    /** SOCKET **/
	int localSocket,
        remoteSocket,
        port = 4097;                               

    struct  sockaddr_in localAddr,
                        remoteAddr;
    pthread_t thread_id;
    
           
    int addrLen = sizeof(struct sockaddr_in);
    
    localSocket = socket(AF_INET , SOCK_STREAM , 0);
    if (localSocket == -1){
         perror("socket FLIR failed!!");
    } 
    
    localAddr.sin_family = AF_INET;
    localAddr.sin_addr.s_addr = INADDR_ANY;
    localAddr.sin_port = htons( port );
    
    if( bind(localSocket,(struct sockaddr *)&localAddr , sizeof(localAddr)) < 0) {
         perror("bind FLIR socket");
         exit(1);
    }
    
    listen(localSocket , 3);
    
    /************/
    
    //img = Mat::zeros(720 , 1280, CV_8UC3);  
    if (!image.isContinuous()) {
        image = image.clone();
    }
    int imgSize = image.total() * image.elemSize();
    int bytes = 0;
    
    
    //Start capture    
	
	while (!app_exit) {
		
		std::cout << "FLIR waiting connection..." << std::endl;
		remoteSocket = accept(localSocket, (struct sockaddr *)&remoteAddr, (socklen_t*)&addrLen);
		
		if (remoteSocket < 0) {
			perror("accept failed!");
			exit(1);
		} 
		else
		{
			while(1)
			{						
#ifdef FLIR			
				if( frameControl == 0 || lastFrame <= frameControl2)
				{
#endif
					written_frames_count++;
					Camera.grab();
					Camera.retrieve ( image);
					
					if ((bytes = send(socket, image.data, imgSize, 0)) < 0){
						 std::cerr << "bytes = " << bytes << std::endl;
						 break; // Não enviou mais, então sai e espera outra conexão
					}

					frame_count++;
					printf("RGB control: %ld\n",frameControl);
					frameControl ++;
					lastFrame++;
				}
#ifdef FLIR			
			}
#endif
		}
	}
	
	printf("Fim RGB!\n");
	
	
	pthread_exit(NULL);
}

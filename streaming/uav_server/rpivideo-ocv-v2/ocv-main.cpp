#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//#define DEBUG

#define FLIR

#include "ocv-gui.h"
#include "ocv-lepton-threads.h"

/*
 * IMPORTANT THINGS:
 * 
 * A Comprehensive Guide to Installing and Configuring OpenCV 2.4.2 on Ubuntu
 * http://www.ozbotz.org/opencv-installation/
 */


int main( int argc, char **argv )
{
#ifdef FLIR	
	
		int x, a, b, c;
		
		a = b = c = 0;
		for(x=0; colormap_rainbow[x] != -1; x++, a++);
		for(x=0; colormap_grayscale[x] != -1; x++, b++);
		for(x=0; colormap_ironblack[x] != -1; x++, c++);
		printf("\n\n%d\t%d\t%d\n\n", a, b, c);
	

		// initialize SPI
		int spi_status = SpiOpenPort(0);
		if (spi_status != 0) {
			printf("Problems with SPI ! [%d]", spi_status);
			exit(-1);
		}

		// creating the mutex to control the concurrent access to FLIRImage
		pthread_mutex_init(&mutex_FLIRImage, NULL);
		pthread_mutex_init(&mutex_flirData, NULL); 

		// creating the FLIR image
		//loadImage("maw.png");
		createFLIRImage(); 

		// creating application threads
		pthread_t thread_readFLIR;
		pthread_create(&thread_readFLIR, NULL, readDataFromFLIR, NULL);

		pthread_t thread_displayFLIRImage;
		pthread_create(&thread_displayFLIRImage, NULL, displayFLIRDataAsImage, NULL);
		

		pthread_t thread_writeFLIRVideo;
		pthread_create(&thread_writeFLIRVideo, NULL, writeFLIRVideo, NULL);
#endif
		pthread_t thread_writeRGBVideo;
		pthread_create(&thread_writeRGBVideo, NULL, writeRGBVideo, NULL);

		
		// infinite loop to avoid the program to finish
		app_exit = 0;
		printf("Aguardando: ");
		while(app_exit)
		{
			int cmd = 0;
			cout << "Command 0 or 1";	
			cin >> cmd;
			
			if(cmd == 0 or cmd == 1)
				app_exit = 1;
			else
				cout << "Invalid! Try again!" << endl;
		}
		app_exit = 1;
		sleep(1);

#ifdef FLIR		
		// wait until all threads finish
	#ifndef DEBUG
		printf("\n\nWaiting for thread 1 to complete...\n");
	#endif
		pthread_join(thread_readFLIR, NULL);
	#ifndef DEBUG
		printf("ok\nWaiting for thread 2 to complete...");
	#endif
		pthread_join(thread_displayFLIRImage, NULL);
	#ifndef DEBUG
		printf("ok\nWaiting for thread 3 to complete...");
	#endif
	

		pthread_join(thread_writeFLIRVideo, NULL);
#endif		
		pthread_join(thread_writeRGBVideo, NULL);
		
		printf("\nok\n");

#ifdef FLIR		
		// releasing allocated resources
		pthread_mutex_destroy(&mutex_FLIRImage);
		pthread_mutex_destroy(&mutex_flirData);
		
		// close spi port
		SpiClosePort(0);
		
			
		if (mwFLIRImage)
			cvReleaseImage(&mwFLIRImage);
#endif
		if (mwImage)
			cvReleaseImage(&mwImage);
	
	printf("\nbye bye !!\n\n\n");
	return 0;
}

